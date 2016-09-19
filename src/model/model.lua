 --[[ Model, adapted from https://github.com/harvardnlp/seq2seq-attn/blob/master/train.lua
--]]
require 'nn'
require 'hdf5'
require 'cudnn'
require 'optim'
require 'paths'

package.path = package.path .. ';src/?.lua' .. ';src/utils/?.lua' .. ';src/model/?.lua' .. ';src/optim/?.lua'
require 'cnn'
require 'LSTM'
require 'output_projector'
require 'criterion'
require 'model_utils'
require 'optim_adadelta'
require 'optim_sgd'
require 'memory'

local model = torch.class('Model')

--[[ Args: 
-- config.load_model
-- config.model_dir
-- config.dropout
-- config.encoder_num_hidden
-- config.encoder_num_layers
-- config.decoder_num_layers
-- config.target_vocab_size
-- config.target_embedding_size
-- config.max_encoder_l_w
-- config.max_decoder_l
-- config.input_feed
-- config.batch_size
--]]

-- init
function model:__init()
    if logging ~= nil then
        log = function(msg) logging:info(msg) end
    else
        log = print
    end
end

-- load model from model_path
function model:load(model_path, config)
    config = config or {}

    -- Build model

    assert(paths.filep(model_path), string.format('Model %s does not exist!', model_path))

    local checkpoint = torch.load(model_path)
    local model, model_config = checkpoint[1], checkpoint[2]
    preallocateMemory(model_config.prealloc)
    self.cnn_model = model[1]:double()
    self.encoder_fw = model[2]:double()
    self.encoder_bw = model[3]:double()
    self.decoder = model[4]:double()      
    self.output_projector = model[5]:double()
    self.pos_embedding_fw = model[6]:double()
    self.pos_embedding_bw = model[7]:double()
    self.global_step = checkpoint[3]
    self.optim_state = checkpoint[4]
    id2vocab = checkpoint[5]

    -- Load model structure parameters
    self.cnn_feature_size = 512
    self.dropout = model_config.dropout
    self.encoder_num_hidden = model_config.encoder_num_hidden
    self.encoder_num_layers = model_config.encoder_num_layers
    self.decoder_num_hidden = self.encoder_num_hidden * 2
    self.decoder_num_layers = model_config.decoder_num_layers
    self.target_vocab_size = #id2vocab+4
    self.target_embedding_size = model_config.target_embedding_size
    self.input_feed = model_config.input_feed
    self.prealloc = model_config.prealloc

    self.max_encoder_l_w = config.max_encoder_l_w or model_config.max_encoder_l_w
    self.max_encoder_l_h = config.max_encoder_l_h or model_config.max_encoder_l_h
    self.max_decoder_l = config.max_decoder_l or model_config.max_decoder_l
    self.batch_size = config.batch_size or model_config.batch_size

    if config.max_encoder_l_h > model_config.max_encoder_l_h then
        local pos_embedding_fw = nn.Sequential():add(nn.LookupTable(self.max_encoder_l_h,self.encoder_num_layers*self.encoder_num_hidden*2))
        local pos_embedding_bw = nn.Sequential():add(nn.LookupTable(self.max_encoder_l_h, self.encoder_num_layers*self.encoder_num_hidden*2))
        for i = 1, self.max_encoder_l_h do
            local j = math.min(i, model_config.max_encoder_l_h)
            pos_embedding_fw.get(1).weight[i] = self.pos_embedding_fw.get(1).weight[j]
            pos_embedding_bw.get(1).weight[i] = self.pos_embedding_bw.get(1).weight[j]
        end
    end
    self:_build()
end

-- create model with fresh parameters
function model:create(config)
    self.cnn_feature_size = 512
    self.dropout = config.dropout
    self.encoder_num_hidden = config.encoder_num_hidden
    self.encoder_num_layers = config.encoder_num_layers
    self.decoder_num_hidden = config.encoder_num_hidden * 2
    self.decoder_num_layers = config.decoder_num_layers
    self.target_vocab_size = config.target_vocab_size
    self.target_embedding_size = config.target_embedding_size
    self.max_encoder_l_w = config.max_encoder_l_w
    self.max_encoder_l_h = config.max_encoder_l_h
    self.max_decoder_l = config.max_decoder_l
    self.input_feed = config.input_feed
    self.batch_size = config.batch_size
    self.prealloc = config.prealloc
    preallocateMemory(config.prealloc)

    self.pos_embedding_fw = nn.Sequential():add(nn.LookupTable(self.max_encoder_l_h,self.encoder_num_layers*self.encoder_num_hidden*2))
    self.pos_embedding_bw = nn.Sequential():add(nn.LookupTable(self.max_encoder_l_h, self.encoder_num_layers*self.encoder_num_hidden*2))
    -- CNN model, input size: (batch_size, 1, 32, width), output size: (batch_size, sequence_length, 512)
    self.cnn_model = createCNNModel()
    -- createLSTM(input_size, num_hidden, num_layers, dropout, use_attention, input_feed, use_lookup, vocab_size)
    self.encoder_fw = createLSTM(self.cnn_feature_size, self.encoder_num_hidden, self.encoder_num_layers, self.dropout, false, false, false, nil, self.batch_size, self.max_encoder_l, 'encoder-fw')
    self.encoder_bw = createLSTM(self.cnn_feature_size, self.encoder_num_hidden, self.encoder_num_layers, self.dropout, false, false, false, nil, self.batch_size, self.max_encoder_l, 'encoder-bw')
    self.decoder = createLSTM(self.target_embedding_size, self.decoder_num_hidden, self.decoder_num_layers, self.dropout, true, self.input_feed, true, self.target_vocab_size, self.batch_size, self.max_encoder_l, 'decoder')
    self.output_projector = createOutputUnit(self.decoder_num_hidden, self.target_vocab_size)
    self.global_step = 0
    self._init = true

    self.optim_state = {}
    self.optim_state.learningRate = config.learning_rate
    self:_build()
end

-- build
function model:_build()
    log(string.format('cnn_featuer_size: %d', self.cnn_feature_size))
    log(string.format('dropout: %f', self.dropout))
    log(string.format('encoder_num_hidden: %d', self.encoder_num_hidden))
    log(string.format('encoder_num_layers: %d', self.encoder_num_layers))
    log(string.format('decoder_num_hidden: %d', self.decoder_num_hidden))
    log(string.format('decoder_num_layers: %d', self.decoder_num_layers))
    log(string.format('target_vocab_size: %d', self.target_vocab_size))
    log(string.format('target_embedding_size: %d', self.target_embedding_size))
    log(string.format('max_encoder_l_w: %d', self.max_encoder_l_w))
    log(string.format('max_decoder_l: %d', self.max_decoder_l))
    log(string.format('input_feed: %s', self.input_feed))
    log(string.format('batch_size: %d', self.batch_size))
    log(string.format('prealloc: %s', self.prealloc))


    self.config = {}
    self.config.dropout = self.dropout
    self.config.encoder_num_hidden = self.encoder_num_hidden
    self.config.encoder_num_layers = self.encoder_num_layers
    self.config.decoder_num_hidden = self.decoder_num_hidden
    self.config.decoder_num_layers = self.decoder_num_layers
    self.config.target_vocab_size = self.target_vocab_size
    self.config.target_embedding_size = self.target_embedding_size
    self.config.max_encoder_l_w = self.max_encoder_l_w
    self.config.max_encoder_l_h = self.max_encoder_l_h
    self.config.max_decoder_l = self.max_decoder_l
    self.config.input_feed = self.input_feed
    self.config.batch_size = self.batch_size
    self.config.prealloc = self.prealloc


    if self.optim_state == nil then
        self.optim_state = {}
    end
    self.criterion = createCriterion(self.target_vocab_size)

    -- convert to cuda if use gpu
    self.layers = {self.cnn_model, self.encoder_fw, self.encoder_bw, self.decoder, self.output_projector, self.pos_embedding_fw, self.pos_embedding_bw}
    for i = 1, #self.layers do
        localize(self.layers[i])
    end
    localize(self.criterion)

    self.context_proto = localize(torch.zeros(self.batch_size, self.max_encoder_l_w*self.max_encoder_l_h, 2*self.encoder_num_hidden))
    self.encoder_fw_grad_proto = localize(torch.zeros(self.batch_size, self.max_encoder_l_w*self.max_encoder_l_h, self.encoder_num_hidden))
    self.encoder_bw_grad_proto = localize(torch.zeros(self.batch_size, self.max_encoder_l_w*self.max_encoder_l_h, self.encoder_num_hidden))
    self.cnn_grad_proto = localize(torch.zeros(self.max_encoder_l_h, self.batch_size, self.max_encoder_l_w, self.cnn_feature_size))
    self.pos_embedding_grad_fw_proto = localize(torch.zeros(self.batch_size, self.encoder_num_layers*self.encoder_num_hidden*2))

    local num_params = 0
    self.params, self.grad_params = {}, {}
    for i = 1, #self.layers do
        local p, gp = self.layers[i]:getParameters()
        if self._init then
            p:uniform(-0.05,0.05)
        end
        num_params = num_params + p:size(1)
        self.params[i] = p
        self.grad_params[i] = gp
    end
    log(string.format('Number of parameters: %d', num_params))

    self.decoder_clones = clone_many_times(self.decoder, self.max_decoder_l)
    self.encoder_fw_clones = clone_many_times(self.encoder_fw, self.max_encoder_l_w)
    self.encoder_bw_clones = clone_many_times(self.encoder_bw, self.max_encoder_l_w)
    
    for i = 1, #self.encoder_fw_clones do
        if self.encoder_fw_clones[i].apply then
            self.encoder_fw_clones[i]:apply(function(m) m:setReuse() end)
            if self.prealloc then self.encoder_fw_clones[i]:apply(function(m) m:setPrealloc() end) end
        end
    end
    for i = 1, #self.encoder_bw_clones do
        if self.encoder_bw_clones[i].apply then
            self.encoder_bw_clones[i]:apply(function(m) m:setReuse() end)
            if self.prealloc then self.encoder_bw_clones[i]:apply(function(m) m:setPrealloc() end) end
        end
    end
    for i = 1, #self.decoder_clones do
        if self.decoder_clones[i].apply then
            self.decoder_clones[i]:apply(function (m) m:setReuse() end)
            if self.prealloc then self.decoder_clones[i]:apply(function(m) m:setPrealloc() end) end
        end
    end
    -- initalial states
    local encoder_h_init = localize(torch.zeros(self.batch_size, self.encoder_num_hidden))
    local decoder_h_init = localize(torch.zeros(self.batch_size, self.decoder_num_hidden))

    self.init_fwd_enc = {}
    self.init_bwd_enc = {}
    self.init_fwd_dec = {}
    self.init_bwd_dec = {}
    for L = 1, self.encoder_num_layers do
        table.insert(self.init_fwd_enc, encoder_h_init:clone())
        table.insert(self.init_fwd_enc, encoder_h_init:clone())
        table.insert(self.init_bwd_enc, encoder_h_init:clone())
        table.insert(self.init_bwd_enc, encoder_h_init:clone())
    end
    if self.input_feed then
        table.insert(self.init_fwd_dec, decoder_h_init:clone())
    end
    table.insert(self.init_bwd_dec, decoder_h_init:clone())
    for L = 1, self.decoder_num_layers do
        table.insert(self.init_fwd_dec, decoder_h_init:clone()) -- memory cell
        table.insert(self.init_fwd_dec, decoder_h_init:clone()) -- hidden state
        table.insert(self.init_bwd_dec, decoder_h_init:clone())
        table.insert(self.init_bwd_dec, decoder_h_init:clone()) 
    end
    self.dec_offset = 3 -- offset depends on input feeding
    if self.input_feed then
        self.dec_offset = self.dec_offset + 1
    end
    self.init_beam = false
    self.visualize = false
    collectgarbage()
end

-- one step 
function model:step(batch, forward_only, beam_size, trie)
    if forward_only then
        self.val_batch_size = self.batch_size
        beam_size = beam_size or 1 -- default argmax
        beam_size = math.min(beam_size, self.target_vocab_size)
        if not self.init_beam then
            self.init_beam = true
            local beam_decoder_h_init = localize(torch.zeros(self.val_batch_size*beam_size, self.decoder_num_hidden))
            self.beam_scores = localize(torch.zeros(self.val_batch_size, beam_size))
            self.current_indices_history = {}
            self.beam_parents_history = {}
            self.beam_init_fwd_dec = {}
            if self.input_feed then
                table.insert(self.beam_init_fwd_dec, beam_decoder_h_init:clone())
            end
            for L = 1, self.decoder_num_layers do
                table.insert(self.beam_init_fwd_dec, beam_decoder_h_init:clone()) -- memory cell
                table.insert(self.beam_init_fwd_dec, beam_decoder_h_init:clone()) -- hidden state
            end
            self.trie_locations = {}
        else
            self.beam_scores:zero()
            self.current_indices_history = {}
            self.beam_parents_history = {}
            self.trie_locations = {}
        end
    else
        if self.init_beam then
            self.init_beam = false
            self.trie_locations = {}
            self.beam_init_fwd_dec = {}
            self.current_indices_history = {}
            self.beam_parents_history = {}
            self.trie_locations = {}
            self.beam_scores = nil
            collectgarbage()
        end
    end
    local input_batch = localize(batch[1])
    local target_batch = localize(batch[2])
    local target_eval_batch = localize(batch[3])
    local num_nonzeros = batch[4]
    local img_paths
    if self.visualize then
        img_paths = batch[5]
    end

    local batch_size = input_batch:size()[1]
    local target_l = target_batch:size()[2]

    assert(target_l <= self.max_decoder_l, string.format('max_decoder_l (%d) < target_l (%d)!', self.max_decoder_l, target_l))
    -- if forward only, then re-generate the target batch
    if forward_only then
        local target_batch_new = localize(torch.IntTensor(batch_size, self.max_decoder_l)):fill(1)
        target_batch_new[{{1,batch_size}, {1,target_l}}]:copy(target_batch)
        target_batch = target_batch_new
        local target_eval_batch_new = localize(torch.IntTensor(batch_size, self.max_decoder_l)):fill(1)
        target_eval_batch_new[{{1,batch_size}, {1,target_l}}]:copy(target_eval_batch)
        target_eval_batch = target_eval_batch_new
        target_l = self.max_decoder_l
    end

    if not forward_only then
        self.cnn_model:training()
        self.output_projector:training()
    else
        self.cnn_model:evaluate()
        --self.cnn_model:training()
        self.output_projector:evaluate()
    end

    local feval = function(p) --cut off when evaluate
        target = target_batch:transpose(1,2)
        target_eval = target_eval_batch:transpose(1,2)
        local cnn_output_list = self.cnn_model:forward(input_batch) -- list of (batch_size, W, 512)
        local counter = 1
        local imgH = #cnn_output_list
        local source_l = cnn_output_list[1]:size()[2]
        local context = self.context_proto[{{1, batch_size}, {1, source_l*imgH}}]
        assert(source_l <= self.max_encoder_l_w, string.format('max_encoder_l_w (%d) < source_l (%d)!', self.max_encoder_l_w, source_l))
        for i = 1, imgH do
            if forward_only then
                self.pos_embedding_fw:evaluate()
                self.pos_embedding_bw:evaluate()
            else
                self.pos_embedding_fw:training()
                self.pos_embedding_bw:training()
            end
            local pos = localize(torch.zeros(batch_size)):fill(i)
            local pos_embedding_fw  = self.pos_embedding_fw:forward(pos)
            local pos_embedding_bw  = self.pos_embedding_bw:forward(pos)
            local cnn_output = cnn_output_list[i] --1, imgW, 512
            source = cnn_output:transpose(1,2) -- imgW,1,512
            -- forward encoder
            local rnn_state_enc = reset_state(self.init_fwd_enc, batch_size, 0)
            for l = 1, self.encoder_num_layers do
                rnn_state_enc[0][l*2-1]:copy(pos_embedding_fw[{{},{(l*2-2)*self.encoder_num_hidden+1, (l*2-1)*self.encoder_num_hidden}}])
                rnn_state_enc[0][l*2]:copy(pos_embedding_fw[{{},{(l*2-1)*self.encoder_num_hidden+1, (l*2)*self.encoder_num_hidden}}])
            end
            for t = 1, source_l do
                counter = (i-1)*source_l + t
                if not forward_only then
                    self.encoder_fw_clones[t]:training()
                else
                    self.encoder_fw_clones[t]:evaluate()
                end
                local encoder_input = {source[t], table.unpack(rnn_state_enc[t-1])}
                local out = self.encoder_fw_clones[t]:forward(encoder_input)
                rnn_state_enc[t] = out
                context[{{},counter, {1, self.encoder_num_hidden}}]:copy(out[#out])
            end
            local rnn_state_enc_bwd = reset_state(self.init_fwd_enc, batch_size, source_l+1)
            for l = 1, self.encoder_num_layers do
                rnn_state_enc_bwd[source_l+1][l*2-1]:copy(pos_embedding_bw[{{},{(l*2-2)*self.encoder_num_hidden+1, (l*2-1)*self.encoder_num_hidden}}])
                rnn_state_enc_bwd[source_l+1][l*2]:copy(pos_embedding_bw[{{},{(l*2-1)*self.encoder_num_hidden+1, (l*2)*self.encoder_num_hidden}}])
            end
            for t = source_l, 1, -1 do
                counter = (i-1)*source_l + t
                if not forward_only then
                    self.encoder_bw_clones[t]:training()
                else
                    self.encoder_bw_clones[t]:evaluate()
                end
                local encoder_input = {source[t], table.unpack(rnn_state_enc_bwd[t+1])}
                local out = self.encoder_bw_clones[t]:forward(encoder_input)
                rnn_state_enc_bwd[t] = out
                context[{{},counter, {1+self.encoder_num_hidden, 2*self.encoder_num_hidden}}]:copy(out[#out])
            end
        end
        local preds = {}
        local indices
        local rnn_state_dec
        -- forward_only == true, beam search
        if forward_only then
            local beam_replicate = function(hidden_state)
                if hidden_state:dim() == 1 then
                    local batch_size = hidden_state:size()[1]
                    if not hidden_state:isContiguous() then
                        hidden_state = hidden_state:contiguous()
                    end
                    local temp_state = hidden_state:view(batch_size, 1):expand(batch_size, beam_size)
                    if not temp_state:isContiguous() then
                        temp_state = temp_state:contiguous()
                    end
                    return temp_state:view(-1)
                elseif hidden_state:dim() == 2 then
                    local batch_size = hidden_state:size()[1]
                    local num_hidden = hidden_state:size()[2]
                    if not hidden_state:isContiguous() then
                        hidden_state = hidden_state:contiguous()
                    end
                    local temp_state = hidden_state:view(batch_size, 1, num_hidden):expand(batch_size, beam_size, num_hidden)
                    if not temp_state:isContiguous() then
                        temp_state = temp_state:contiguous()
                    end
                    return temp_state:view(batch_size*beam_size, num_hidden)
                elseif hidden_state:dim() == 3 then
                    local batch_size = hidden_state:size()[1]
                    local source_l = hidden_state:size()[2]
                    local num_hidden = hidden_state:size()[3]
                    if not hidden_state:isContiguous() then
                        hidden_state = hidden_state:contiguous()
                    end
                    local temp_state = hidden_state:view(batch_size, 1, source_l, num_hidden):expand(batch_size, beam_size, source_l, num_hidden)
                    if not temp_state:isContiguous() then
                        temp_state = temp_state:contiguous()
                    end
                    return temp_state:view(batch_size*beam_size, source_l, num_hidden)
                else
                    assert(false, 'does not support ndim except for 2 and 3')
                end
            end
            rnn_state_dec = reset_state(self.beam_init_fwd_dec, batch_size, 0)
            --local L = self.encoder_num_layers
            --if self.input_feed then
            --    rnn_state_dec[0][1*2-1+1]:copy((torch.cat(rnn_state_enc[source_l][L*2-1], rnn_state_enc_bwd[1][L*2-1])))
            --    rnn_state_dec[0][1*2+1]:copy((torch.cat(rnn_state_enc[source_l][L*2], rnn_state_enc_bwd[1][L*2])))
            --else
            --    rnn_state_dec[0][1*2-1+0]:copy((torch.cat(rnn_state_enc[source_l][L*2-1], rnn_state_enc_bwd[1][L*2-1])))
            --    rnn_state_dec[0][1*2+0]:copy((torch.cat(rnn_state_enc[source_l][L*2], rnn_state_enc_bwd[1][L*2])))
            --end
            --for L = 2, self.decoder_num_layers do
            --    rnn_state_dec[0][L*2-1+0]:zero()
            --    rnn_state_dec[0][L*2+0]:zero()
            --end
            local beam_context = beam_replicate(context)
            local decoder_input
            local beam_input
            for t = 1, target_l do
                self.decoder_clones[t]:evaluate()
                if t == 1 then
                    -- self.trie_locations
                    if trie ~= nil then
                        for b = 1, batch_size do
                            if self.trie_locations[b] == nil then
                                self.trie_locations[b] = {}
                            end
                            self.trie_locations[b] = trie[2]
                        end
                    end
                    beam_input = target[t]
                    decoder_input = {beam_input, context, table.unpack(rnn_state_dec[t-1])}
                else
                    decoder_input = {beam_input, beam_context, table.unpack(rnn_state_dec[t-1])}
                end
                local out = self.decoder_clones[t]:forward(decoder_input)
                local next_state = {}
                local top_out = out[#out]
                local probs = self.output_projector:forward(top_out) -- t~=0, batch_size*beam_size, vocab_size; t=0, batch_size,vocab_size
                local current_indices, raw_indices
                local beam_parents
                if t == 1 then
                    -- probs batch_size, vocab_size
                    if trie == nil then
                        self.beam_scores, raw_indices = probs:topk(beam_size, true)
                        raw_indices = localize(raw_indices:double())
                        current_indices = raw_indices
                    else
                        self.beam_scores:zero()
                        raw_indices = localize(torch.zeros(batch_size, beam_size))
                        local _, i = probs:sort(2, true)
                        for b = 1, batch_size do
                            local num_beam = 0
                            for vocab = 1, self.target_vocab_size do
                                if num_beam ~= beam_size then
                                    local vocab_id = i[b][vocab]
                                    if self.trie_locations[b][vocab_id] ~= nil then
                                        num_beam = num_beam + 1
                                        raw_indices[b][num_beam] = vocab_id
                                        self.beam_scores[b][num_beam] = probs[b][vocab_id]
                                    end
                                end
                            end
                            if num_beam ~= beam_size then
                                log(string.format('Warning: valid beam size: %d', num_beam))
                                local vocab_id = nil
                                for vocab = 1, self.target_vocab_size do
                                    if vocab_id == nil then
                                        local vocab_id_tmp = i[b][vocab]
                                        if self.trie_locations[b][vocab_id_tmp] ~= nil then
                                            vocab_id = vocab_id_tmp
                                        end
                                    end
                                end
                                for beam = num_beam+1, beam_size do
                                    raw_indices[b][beam] = vocab_id
                                    self.beam_scores[b][beam] = probs[b][vocab_id]
                                end
                            end
                            local trie_locations = {}
                            for beam = 1, beam_size do
                                local vocab_id = raw_indices[b][beam]
                                trie_locations[beam] = self.trie_locations[b][vocab_id]
                            end
                            self.trie_locations[b] = trie_locations
                        end
                        current_indices = raw_indices
                    end
                else
                    -- batch_size*beam_size, vocab_size
                    probs:select(2,1):maskedFill(beam_input:eq(1), 0) -- once padding or EOS encountered, stuck at that point
                    probs:select(2,1):maskedFill(beam_input:eq(3), 0)
                    local total_scores = (probs:view(batch_size, beam_size, self.target_vocab_size) + self.beam_scores[{{1,batch_size}, {}}]:view(batch_size, beam_size, 1):expand(batch_size, beam_size, self.target_vocab_size)):view(batch_size, beam_size*self.target_vocab_size) -- batch_size, beam_size * target_vocab_size
                    if trie == nil then
                        self.beam_scores, raw_indices = total_scores:topk(beam_size, true) --batch_size, beam_size
                        raw_indices = localize(raw_indices:double())
                        raw_indices:add(-1)
                        if use_cuda then
                            current_indices = raw_indices:double():fmod(self.target_vocab_size):cuda()+1 -- batch_size, beam_size for current vocab
                        else
                            current_indices = raw_indices:fmod(self.target_vocab_size)+1 -- batch_size, beam_size for current vocab
                        end
                    else
                        raw_indices = localize(torch.zeros(batch_size, beam_size))
                        current_indices = localize(torch.zeros(batch_size, beam_size))
                        local _, i = total_scores:sort(2, true) -- batch_size, beam_size*target_size
                        for b = 1, batch_size do
                            local num_beam = 0
                            for beam_vocab = 1, beam_size*self.target_vocab_size do
                                if num_beam ~= beam_size then
                                    local beam_vocab_id = i[b][beam_vocab]
                                    local vocab_id = (beam_vocab_id-1) % self.target_vocab_size + 1
                                    local beam_id = math.floor((beam_vocab_id-1) / self.target_vocab_size)+1 -- batch_size, beam_size for number of beam in each batch
                                    if vocab_id == 1 or self.trie_locations[b][beam_id][vocab_id] ~= nil then
                                        num_beam = num_beam + 1
                                        current_indices[b][num_beam] = vocab_id
                                        raw_indices[b][num_beam] = beam_vocab_id-1
                                        self.beam_scores[b][num_beam] = total_scores[b][beam_vocab_id]
                                    end
                                end
                            end
                            if num_beam ~= beam_size then
                                log(string.format('Warning: valid beam size: %d', num_beam))
                                local beam_vocab_id = nil
                                for beam_vocab = 1, beam_size*self.target_vocab_size do
                                    if beam_vocab_id == nil then
                                        local beam_vocab_id_tmp = i[b][beam_vocab]
                                        local vocab_id = (beam_vocab_id_tmp-1) % self.target_vocab_size + 1
                                        local beam_id = math.floor((beam_vocab_id_tmp-1) / self.target_vocab_size)+1 -- batch_size, beam_size for number of beam in each batch
                                        if vocab_id == 1 or self.trie_locations[b][vocab_id_tmp] ~= nil then
                                            beam_vocab_id = vocab_id_tmp
                                        end
                                    end
                                end
                                for beam = num_beam+1, beam_size do
                                    local vocab_id = (beam_vocab_id-1) % self.target_vocab_size + 1
                                    local beam_id = ((beam_vocab_id-1) / self.target_vocab_size):floor()+1 -- batch_size, beam_size for number of beam in each batch
                                    current_indices[b][beam] = vocab_id
                                    raw_indices[b][beam] = beam_vocab_id-1
                                    self.beam_scores[b][beam] = total_scores[b][beam_vocab_id]
                                end
                            end
                            local trie_locations = {}
                            for beam = 1, beam_size do
                                local beam_vocab_id = raw_indices[b][beam]
                                local beam_id = math.floor((beam_vocab_id) / self.target_vocab_size)+1 -- batch_size, beam_size for number of beam in each batch
                                local vocab_id = (beam_vocab_id) % self.target_vocab_size + 1
                                if vocab_id == 1 then
                                    trie_locations[beam] = self.trie_locations[b][beam_id]
                                else
                                    trie_locations[beam] = self.trie_locations[b][beam_id][vocab_id]
                                end
                            end
                            self.trie_locations[b] = trie_locations
                        end
                        
                    end
                end
                beam_parents = localize(raw_indices:int()/self.target_vocab_size+1) -- batch_size, beam_size for number of beam in each batch
                beam_input = current_indices:view(batch_size*beam_size)
                table.insert(self.current_indices_history, current_indices:clone())
                table.insert(self.beam_parents_history, beam_parents:clone())

                if self.input_feed then
                    local top_out = out[#out] -- batch_size*beam_size, hidden_dim
                    if t == 1 then
                        top_out = beam_replicate(top_out)
                    end
                    table.insert(next_state, top_out:index(1, beam_parents:view(-1)+localize(torch.range(0,(batch_size-1)*beam_size,beam_size):long()):contiguous():view(batch_size,1):expand(batch_size,beam_size):contiguous():view(-1)))
                end
                for j = 1, #out-1 do
                    local out_j = out[j] -- batch_size*beam_size, hidden_dim
                    if t == 1 then
                        out_j = beam_replicate(out_j)
                    end
                    table.insert(next_state, out_j:index(1, beam_parents:view(-1)+localize(torch.range(0,(batch_size-1)*beam_size,beam_size):long()):contiguous():view(batch_size,1):expand(batch_size,beam_size):contiguous():view(-1)))
                end
                rnn_state_dec[t] = next_state
            end
        else -- forward_only == false
            -- set decoder states
            rnn_state_dec = reset_state(self.init_fwd_dec, batch_size, 0)
            -- not use encoder final state to initialize the first layer
            -- local L = self.encoder_num_layers
            --if self.input_feed then
            --    rnn_state_dec[0][1*2-1+1]:copy(torch.cat(rnn_state_enc[source_l][L*2-1], rnn_state_enc_bwd[1][L*2-1]))
            --    rnn_state_dec[0][1*2+1]:copy(torch.cat(rnn_state_enc[source_l][L*2], rnn_state_enc_bwd[1][L*2]))
            --else
            --    rnn_state_dec[0][1*2-1+0]:copy(torch.cat(rnn_state_enc[source_l][L*2-1], rnn_state_enc_bwd[1][L*2-1]))
            --    rnn_state_dec[0][1*2+0]:copy(torch.cat(rnn_state_enc[source_l][L*2], rnn_state_enc_bwd[1][L*2]))
            --end
            --for L = 2, self.decoder_num_layers do
            --    rnn_state_dec[0][L*2-1+0]:zero()
            --    rnn_state_dec[0][L*2+0]:zero()
            --end
            for t = 1, target_l do
                self.decoder_clones[t]:training()
                local decoder_input
                decoder_input = {target[t], context, table.unpack(rnn_state_dec[t-1])}
                local out = self.decoder_clones[t]:forward(decoder_input)
                local next_state = {}
                table.insert(preds, out[#out])
                if self.input_feed then
                    table.insert(next_state, out[#out])
                end
                for j = 1, #out-1 do
                    table.insert(next_state, out[j])
                end
                rnn_state_dec[t] = next_state
            end
        end
        local loss, accuracy = 0.0, 0.0
        if forward_only then
            -- final decoding
            local labels = localize(torch.zeros(batch_size, target_l)):fill(1)
            local scores, indices = torch.max(self.beam_scores[{{1,batch_size},{}}], 2) -- batch_size, 1
            indices = localize(indices:double())
            scores = scores:view(-1) -- batch_size
            indices = indices:view(-1) -- batch_size
            local current_indices = self.current_indices_history[#self.current_indices_history]:view(-1):index(1,indices+localize(torch.range(0,(batch_size-1)*beam_size, beam_size):long())) --batch_size
            for t = target_l, 1, -1 do
                labels[{{1,batch_size}, t}]:copy(current_indices)
                indices = self.beam_parents_history[t]:view(-1):index(1,indices+localize(torch.range(0,(batch_size-1)*beam_size, beam_size):long())) --batch_size
                if t > 1 then
                    current_indices = self.current_indices_history[t-1]:view(-1):index(1,indices+localize(torch.range(0,(batch_size-1)*beam_size, beam_size):long())) --batch_size
                end
            end
            local word_err, labels_pred, labels_gold, labels_list_pred, labels_list_gold = evalHTMLErrRate(labels, target_eval_batch, self.visualize)
            accuracy = batch_size - word_err
            if self.visualize then
                -- get gold score
                rnn_state_dec = reset_state(self.init_fwd_dec, batch_size, 0)
                local gold_scores = localize(torch.zeros(batch_size))
                for t = 1, target_l do
                    self.decoder_clones[t]:evaluate()
                    local decoder_input
                    decoder_input = {target[t], context, table.unpack(rnn_state_dec[t-1])}
                    local out = self.decoder_clones[t]:forward(decoder_input)
                    local next_state = {}
                    local pred = self.output_projector:forward(out[#out]) --batch_size, vocab_size
                    for j = 1, batch_size do
                        if target_eval[t][j] ~= 1 then
                            gold_scores[j] = gold_scores[j] + pred[j][target_eval[t][j]]
                        end
                    end

                    if self.input_feed then
                        table.insert(next_state, out[#out])
                    end
                    for j = 1, #out-1 do
                        table.insert(next_state, out[j])
                    end
                    rnn_state_dec[t] = next_state
                end
                -- use predictions to visualize attns
                local attn_probs = localize(torch.zeros(batch_size, target_l, source_l*imgH))
                local attn_positions_h = localize(torch.zeros(batch_size, target_l))
                local attn_positions_w = localize(torch.zeros(batch_size, target_l))
                rnn_state_dec = reset_state(self.init_fwd_dec, batch_size, 0)
                for t = 1, target_l do
                    self.decoder_clones[t]:evaluate()
                    local decoder_input
                    if t == 1 then
                        decoder_input = {target[t], context, table.unpack(rnn_state_dec[t-1])}
                    else
                        decoder_input = {labels[{{1,batch_size},t-1}], context, table.unpack(rnn_state_dec[t-1])}
                    end
                    local out = self.decoder_clones[t]:forward(decoder_input)
                    -- print attn
                    attn_probs[{{}, t, {}}]:copy(self.softmax_attn_clones[t].output)
                    local _, attn_inds = torch.max(self.softmax_attn_clones[t].output, 2) --batch_size, 1
                    attn_inds = attn_inds:view(-1) --batch_size
                    for kk = 1, batch_size do
                        local counter = attn_inds[kk]
                        local p_i = math.floor((counter-1) / source_l) + 1
                        local p_t = counter-1 - (p_i-1)*source_l + 1
                        attn_positions_h[kk][t] = p_i
                        attn_positions_w[kk][t] = p_t
                        --print (string.format('%d, %d', p_i, p_t))
                    end
                        --for kk = 1, fea_inds:size(1) do
                    local next_state = {}
                    --table.insert(preds, out[#out])
                    local pred = self.output_projector:forward(out[#out]) --batch_size, vocab_size
                    -- target_eval[t] --batch_size
                    --for j = 1, batch_size do
                    --    if target_eval[t][j] ~= 1 then
                    --        gold_scores[j] = gold_scores[j] + pred[j][target_eval[t][j]]
                    --    end
                    --end

                    if self.input_feed then
                        table.insert(next_state, out[#out])
                    end
                    for j = 1, #out-1 do
                        table.insert(next_state, out[j])
                    end
                    rnn_state_dec[t] = next_state
                end
                for i = 1, #img_paths do
                    self.visualize_file:write(string.format('%s\t%s\t%s\t%f\t%f\t\n', img_paths[i], labels_gold[i], labels_pred[i], scores[i], gold_scores[i]))
                    --for j = 1, target_l do
                    --    if labels[i][j] == 3 then
                    --        break
                    --    end
                    --    self.visualize_attn_file:write(string.format('%s\t%d\t%d\t', labels_list_pred[i][j], attn_positions_h[i][j], attn_positions_w[i][j]))
                    --    for k = 1, source_l*imgH do
                    --        self.visualize_attn_file:write(string.format('%f\t', attn_probs[i][j][k]))
                    --    end
                    --end
                    --self.visualize_attn_file:write('\n')
                end
                self.visualize_file:flush()
            end
                --self.visualize_attn_file:flush()
            --else -- forward_only and not visualize
                -- get gold score
                rnn_state_dec = reset_state(self.init_fwd_dec, batch_size, 0)
                local gold_scores = localize(torch.zeros(batch_size))
                for t = 1, target_l do
                    self.decoder_clones[t]:evaluate()
                    local decoder_input
                    decoder_input = {target[t], context, table.unpack(rnn_state_dec[t-1])}
                    local out = self.decoder_clones[t]:forward(decoder_input)
                    local next_state = {}
                    local pred = self.output_projector:forward(out[#out]) --batch_size, vocab_size
                    loss = loss + self.criterion:forward(pred, target_eval[t])/batch_size

                    if self.input_feed then
                        table.insert(next_state, out[#out])
                    end
                    for j = 1, #out-1 do
                        table.insert(next_state, out[j])
                    end
                    rnn_state_dec[t] = next_state
                end
        else
            local encoder_fw_grads = self.encoder_fw_grad_proto[{{1, batch_size}, {1, source_l*imgH}}]
            local encoder_bw_grads = self.encoder_bw_grad_proto[{{1, batch_size}, {1, source_l*imgH}}]
            for i = 1, #self.grad_params do
                self.grad_params[i]:zero()
            end
            encoder_fw_grads:zero()
            encoder_bw_grads:zero()
            local drnn_state_dec = reset_state(self.init_bwd_dec, batch_size)
            for t = target_l, 1, -1 do
                local pred = self.output_projector:forward(preds[t])
                loss = loss + self.criterion:forward(pred, target_eval[t])/batch_size
                local dl_dpred = self.criterion:backward(pred, target_eval[t])
                dl_dpred:div(batch_size)
                local dl_dtarget = self.output_projector:backward(preds[t], dl_dpred)
                drnn_state_dec[#drnn_state_dec]:add(dl_dtarget)
                local decoder_input = {target[t], context, table.unpack(rnn_state_dec[t-1])}
                local dlst = self.decoder_clones[t]:backward(decoder_input, drnn_state_dec)
                encoder_fw_grads:add(dlst[2][{{}, {}, {1,self.encoder_num_hidden}}])
                encoder_bw_grads:add(dlst[2][{{}, {}, {self.encoder_num_hidden+1, 2*self.encoder_num_hidden}}])
                drnn_state_dec[#drnn_state_dec]:zero()
                if self.input_feed then
                    drnn_state_dec[#drnn_state_dec]:copy(dlst[3])
                end     
                for j = self.dec_offset, #dlst do
                    drnn_state_dec[j-self.dec_offset+1]:copy(dlst[j])
                end
            end
            local cnn_grad = self.cnn_grad_proto[{{1,imgH}, {1,batch_size}, {1,source_l}, {}}]
            -- forward directional encoder
            for i = 1, imgH do
                local cnn_output = cnn_output_list[i]
                source = cnn_output:transpose(1,2) -- 128,1,512
                assert (source_l == cnn_output:size()[2])
                local drnn_state_enc = reset_state(self.init_bwd_enc, batch_size)
                local pos = localize(torch.zeros(batch_size)):fill(i)
                local pos_embedding_fw = self.pos_embedding_fw:forward(pos)
                local pos_embedding_bw = self.pos_embedding_bw:forward(pos)
                --local L = self.encoder_num_layers
                --drnn_state_enc[L*2-1]:copy(drnn_state_dec[1*2-1][{{}, {1, self.encoder_num_hidden}}])
                --drnn_state_enc[L*2]:copy(drnn_state_dec[1*2][{{}, {1, self.encoder_num_hidden}}])
                -- forward encoder
                local rnn_state_enc = reset_state(self.init_fwd_enc, batch_size, 0)
                for l = 1, self.encoder_num_layers do
                    rnn_state_enc[0][l*2-1]:copy(pos_embedding_fw[{{},{(l*2-2)*self.encoder_num_hidden+1, (l*2-1)*self.encoder_num_hidden}}])
                    rnn_state_enc[0][l*2]:copy(pos_embedding_fw[{{},{(l*2-1)*self.encoder_num_hidden+1, (l*2)*self.encoder_num_hidden}}])
                end
                for t = 1, source_l do
                    if not forward_only then
                        self.encoder_fw_clones[t]:training()
                    else
                        self.encoder_fw_clones[t]:evaluate()
                    end
                    local encoder_input = {source[t], table.unpack(rnn_state_enc[t-1])}
                    local out = self.encoder_fw_clones[t]:forward(encoder_input)
                    rnn_state_enc[t] = out
                end
                local rnn_state_enc_bwd = reset_state(self.init_fwd_enc, batch_size, source_l+1)
                for l = 1, self.encoder_num_layers do
                    rnn_state_enc_bwd[source_l+1][l*2-1]:copy(pos_embedding_bw[{{},{(l*2-2)*self.encoder_num_hidden+1, (l*2-1)*self.encoder_num_hidden}}])
                    rnn_state_enc_bwd[source_l+1][l*2]:copy(pos_embedding_bw[{{},{(l*2-1)*self.encoder_num_hidden+1, (l*2)*self.encoder_num_hidden}}])
                end
                for t = source_l, 1, -1 do
                    if not forward_only then
                        self.encoder_bw_clones[t]:training()
                    else
                        self.encoder_bw_clones[t]:evaluate()
                    end
                    local encoder_input = {source[t], table.unpack(rnn_state_enc_bwd[t+1])}
                    local out = self.encoder_bw_clones[t]:forward(encoder_input)
                    rnn_state_enc_bwd[t] = out
                end
                local pos_embedding_grad = self.pos_embedding_grad_fw_proto[{{1,batch_size}}]
                for t = source_l, 1, -1 do
                    counter = (i-1)*source_l + t
                    local encoder_input = {source[t], table.unpack(rnn_state_enc[t-1])}
                    drnn_state_enc[#drnn_state_enc]:add(encoder_fw_grads[{{},counter}])
                    local dlst = self.encoder_fw_clones[t]:backward(encoder_input, drnn_state_enc)
                    for j = 1, #drnn_state_enc do
                        drnn_state_enc[j]:copy(dlst[j+1])
                    end
                    cnn_grad[{i, {}, t, {}}]:copy(dlst[1])
                end
                for l = 1, self.encoder_num_layers do
                    pos_embedding_grad[{{}, {(l*2-2)*self.encoder_num_hidden+1, (l*2-1)*self.encoder_num_hidden}}]:copy(drnn_state_enc[l*2-1])
                    pos_embedding_grad[{{}, {(l*2-1)*self.encoder_num_hidden+1, (l*2)*self.encoder_num_hidden}}]:copy(drnn_state_enc[l*2])
                end
                self.pos_embedding_fw:backward(pos, pos_embedding_grad)
                -- backward directional encoder
                local drnn_state_enc = reset_state(self.init_bwd_enc, batch_size)
                --local L = self.encoder_num_layers
                --drnn_state_enc[L*2-1]:copy(drnn_state_dec[1*2-1][{{}, {self.encoder_num_hidden+1, 2*self.encoder_num_hidden}}])
                --drnn_state_enc[L*2]:copy(drnn_state_dec[1*2][{{}, {self.encoder_num_hidden+1, 2*self.encoder_num_hidden}}])
                for t = 1, source_l do
                    counter = (i-1)*source_l + t
                    local encoder_input = {source[t], table.unpack(rnn_state_enc_bwd[t+1])}
                    drnn_state_enc[#drnn_state_enc]:add(encoder_bw_grads[{{},counter}])
                    local dlst = self.encoder_bw_clones[t]:backward(encoder_input, drnn_state_enc)
                    for j = 1, #drnn_state_enc do
                        drnn_state_enc[j]:copy(dlst[j+1])
                    end
                    --cnn_grad[{{}, t, {}}]:add(dlst[1])
                    cnn_grad[{i, {}, t, {}}]:add(dlst[1])
                end
                local pos_embedding_grad = self.pos_embedding_grad_fw_proto[{{1,batch_size}}]
                for l = 1, self.encoder_num_layers do
                    pos_embedding_grad[{{}, {(l*2-2)*self.encoder_num_hidden+1, (l*2-1)*self.encoder_num_hidden}}]:copy(drnn_state_enc[l*2-1])
                    pos_embedding_grad[{{}, {(l*2-1)*self.encoder_num_hidden+1, (l*2)*self.encoder_num_hidden}}]:copy(drnn_state_enc[l*2])
                end
                self.pos_embedding_bw:backward(pos, pos_embedding_grad)
            end
            -- cnn
            local cnn_final_grad = cnn_grad:split(1, 1)
            for i = 1, #cnn_final_grad do
                cnn_final_grad[i] = cnn_final_grad[i]:contiguous():view(batch_size, source_l, -1)
            end
            self.cnn_model:backward(input_batch, cnn_final_grad)
            collectgarbage()
        end
        return loss, self.grad_params, {num_nonzeros, accuracy}
    end
    local optim_state = self.optim_state
    if not forward_only then
        local _, loss, stats = optim.sgd_list(feval, self.params, optim_state); loss = loss[1]
        return loss*batch_size, stats
    else
        local loss, _, stats = feval(self.params)
        return loss*batch_size, stats -- todo: accuracy
    end
end
-- Set visualize phase
function model:vis(output_dir)
    self.visualize = true
    self.visualize_path = paths.concat(output_dir, 'results.txt')
    self.visualize_attn_path = paths.concat(output_dir, 'results_attn.txt')
    local file, err = io.open(self.visualize_path, "w")
    local file_attn, err_attn = io.open(self.visualize_attn_path, "w")
    self.visualize_file = file
    self.visualize_attn_file = file_attn
    if err then 
        log(string.format('Error: visualize file %s cannot be created', self.visualize_path))
        self.visualize  = false
        self.visualize_file = nil
    elseif err_attn then
        log(string.format('Error: visualize attention file %s cannot be created', self.visualize_attn_path))
        self.visualize  = false
        self.visualize_attn_file = nil
    else
        self.softmax_attn_clones = {}
        for i = 1, #self.decoder_clones do
            local decoder = self.decoder_clones[i]
            local decoder_attn
            decoder:apply(function (layer) 
                if layer.name == 'decoder_attn' then
                    decoder_attn = layer
                end
            end)
            assert (decoder_attn)
            decoder_attn:apply(function (layer)
                if layer.name == 'softmax_attn' then
                    self.softmax_attn_clones[i] = layer
                end
            end)
            assert (self.softmax_attn_clones[i])
        end
    end
end
-- Save model to model_path
function model:save(model_path)
    for i = 1, #self.layers do
        self.layers[i]:clearState()
    end
    torch.save(model_path, {{self.cnn_model, self.encoder_fw, self.encoder_bw, self.decoder, self.output_projector, self.pos_embedding_fw, self.pos_embedding_bw}, self.config, self.global_step, self.optim_state, id2vocab})
end

function model:shutdown()
    if self.visualize_file then
        self.visualize_file:close()
    end
    if self.visualize_attn_file then
        self.visualize_attn_file:close()
    end
end
