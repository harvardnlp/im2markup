# adapted from https://github.com/steerapi/seq2seq-show-att-tell/blob/master/visualizer/generate_html.py

import os, sys, copy, argparse, shutil, pickle, subprocess
import numpy as np
import PIL
from PIL import Image
from utils import run
sys.path.insert(0, '%s'%os.path.join(os.path.dirname(__file__), '../evaluation'))
from img_edit_distance import img_edit_distance_file
TIMEOUT = 10

PAD_TOP = 4*2
PAD_BOTTOM = 4*2
PAD_LEFT = 4*2
PAD_RIGHT = 4*2

def render_and_crop(label, output_path, size):
    script_path = os.path.realpath(__file__)
    script_dir = os.path.dirname(script_path)
    app_dir = os.path.join(script_dir, '../..')
    img_dir = os.path.dirname(output_path)

    line_tmp = '0\t'+label
    file_tmp = os.path.join(os.path.dirname(output_path),'.tmp')
    with open(file_tmp, 'w') as ftmp:
        ftmp.write(line_tmp)
    run('cat %s | python %s'%(file_tmp, os.path.join(app_dir, 'bak_render_latex.py')), TIMEOUT)
    os.remove(file_tmp)
    if not os.path.exists('out0.png'):
        white_path = os.path.join(script_dir, 'white.png')
        white_path_new = os.path.join(img_dir, 'white.png')
        if not os.path.exists(white_path_new):
            shutil.copy(white_path, white_path_new)
        return white_path_new, None
    else:
        old_im = Image.open('out0.png').convert('L')
        img_data = np.asarray(old_im, dtype=np.uint8) # height, width
        nnz_inds = np.where(img_data!=255)
        if len(nnz_inds[0]) == 0:
            white_path = os.path.join(script_dir, 'white.png')
            white_path_new = os.path.join(img_dir, 'white.png')
            if not os.path.exists(white_path_new):
                shutil.copy(white_path, white_path_new)
            return white_path_new, None
        y_min = np.min(nnz_inds[0])
        y_max = np.max(nnz_inds[0])
        x_min = np.min(nnz_inds[1])
        x_max = np.max(nnz_inds[1])
        old_im = old_im.crop((x_min, y_min, x_max+1, y_max+1))
        old_pred_size = old_im.size
        #if not size:
        #    size = (old_pred_size[0]+PAD_LEFT+PAD_RIGHT, old_pred_size[1]+PAD_TOP+PAD_BOTTOM)
        #if old_pred_size[0]+PAD_LEFT+PAD_RIGHT > size[0]:
        #    old_im = old_im.crop((0,0,size[0]-PAD_LEFT-PAD_RIGHT, old_pred_size[1]))
        #if old_pred_size[1]+PAD_TOP+PAD_BOTTOM > size[1]:
        #    old_im = old_im.crop((0,0,old_im.size[0], size[1]-PAD_TOP-PAD_BOTTOM))
        #new_im = Image.new("RGB", size, (255,255,255))   ## luckily, this is already black!
        #new_im.paste(old_im, (PAD_LEFT,PAD_TOP))

        if not size:
            size = (old_pred_size[0], old_pred_size[1])
        if old_pred_size[0] > size[0]:
            old_im = old_im.crop((0,0,size[0], old_pred_size[1]))
        if old_pred_size[1] > size[1]:
            old_im = old_im.crop((0,0,old_im.size[0], size[1]))
        new_im = Image.new("RGB", size, (255,255,255))   ## luckily, this is already black!
        new_im.paste(old_im, (0,0))

        new_im.save(output_path)
        os.remove('out0.png')
        os.remove('out0.pdf')
        os.remove('out0.tex')
        return output_path, size

def main(arguments):
    script_path = os.path.realpath(__file__)
    script_dir = os.path.dirname(script_path)
    app_dir = os.path.join(script_dir, '../..')
    html_head_path = os.path.join(script_dir, 'visualizer.html.template.head')
    assert os.path.exists(html_head_path), 'HTML template %s not found'%html_head_path
    html_tail_path = os.path.join(script_dir, 'visualizer.html.template.tail')
    assert os.path.exists(html_tail_path), 'HTML template %s not found'%html_tail_path

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', dest='output_dir', default='results', help=("Output directory containing results.txt"),
                        type=str)
    parser.add_argument('--data_path', dest='data_path', default='data/im2latex_test_large_filter.lst', help=(""),
                        type=str)
    parser.add_argument('--label_path', dest='label_path', default='data/im2latex_formulas.lst', help=(""),
                        type=str)
 
    args = parser.parse_args(arguments)

    metrics = {}
    metrics['full'] = {}
    metrics['full']['total_length'] = 0
    metrics['full']['correct_length'] = 0
    metrics['full']['num_correct'] = 0
    metrics['full']['num_total'] = 0
    metrics['eliminate'] = {}
    metrics['eliminate']['total_length'] = 0
    metrics['eliminate']['correct_length'] = 0
    metrics['eliminate']['num_correct'] = 0
    metrics['eliminate']['num_total'] = 0

    output_dir = args.output_dir
    data_path = args.data_path
    label_path = args.label_path

    labels_tmp = {}
    labels = {}
    with open(label_path) as flabel:
        with open(data_path) as fdata:
            line_idx = 0
            for line in flabel:
                labels_tmp[line_idx] = line.strip()
                line_idx += 1
            for line in fdata:
                img_path, idx = line.strip().split()
                labels[img_path] = labels_tmp[int(idx)]

    result_path = os.path.join(output_dir, 'results.txt')
    assert os.path.exists(result_path), 'Result file %s not found'%result_path

    website_dir = os.path.join(output_dir, 'website')
    if not os.path.exists(website_dir):
        os.makedirs(website_dir)

    html_path = os.path.join(website_dir, 'index.html')

    img_dir = os.path.join(website_dir, 'images')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    fpred = open(os.path.join(output_dir, 'pred.txt'), 'w')
    fgold = open(os.path.join(output_dir, 'gold.txt'), 'w')
    with open(result_path) as fin:
        with open(html_head_path) as fhead:
            with open(html_tail_path) as ftail:
                with open(html_path, 'w') as fout:
                    for line in fhead:
                        fout.write(line)
                    fout.write('\n')
                    line_idx = 0
                    for line in fin:
                        line_idx += 1
                        print (line_idx)
                        items = line.strip().split('\t')
                        if len(items) == 5:
                            img_path, label_gold, label_pred, score_pred, score_gold = items
                            assert (img_path in labels)
                            fpred.write(label_pred+'\n')
                            fgold.write(label_gold+'\n')
                            img_base_name = os.path.basename(img_path)
                            img_new_path = os.path.join(img_dir, img_base_name)
                            img_orig_path = img_new_path[:-4] + '-orig.png'
                            img_new_orig_path, orig_size = render_and_crop(labels[img_path], img_orig_path, None)
                            img_rel_orig_path = os.path.relpath(img_new_orig_path, website_dir)
                            img_pre_path, pre_size = render_and_crop(label_gold, img_new_path, None)
                            img_rel_path = os.path.relpath(img_pre_path, website_dir)
                            img_pred_path = img_new_path[:-4] + '-pred.png'
                            img_new_pred_path, new_size = render_and_crop(label_pred, img_pred_path, None)
                            img_rel_pred_path = os.path.relpath(img_new_pred_path, website_dir)
                            img_diff_path = img_new_path[:-4] + '-diff.png'
                            img_rel_diff_path = os.path.relpath(img_diff_path, website_dir)

                            
                            edit_distance, ref, edit_distance_eliminate, ref_eliminate, match = img_edit_distance_file(os.path.join(website_dir,img_rel_orig_path), os.path.join(website_dir,img_rel_pred_path), os.path.join(website_dir, img_rel_diff_path))
                            if orig_size:
                                #assert (ref == orig_size[0])
                                if not new_size:
                                    metrics['full']['total_length'] += ref
                                    metrics['full']['num_total'] += 1
                                    metrics['eliminate']['total_length'] += ref_eliminate
                                    metrics['eliminate']['num_total'] += 1
                                else:
                                    metrics['full']['total_length'] += ref
                                    metrics['full']['correct_length'] += (ref-edit_distance)
                                    metrics['full']['num_total'] += 1
                                    #if edit_distance == 0:
                                    if match:
                                        metrics['full']['num_correct'] += 1
                                    metrics['eliminate']['total_length'] += ref_eliminate
                                    metrics['eliminate']['correct_length'] += (ref_eliminate-edit_distance_eliminate)
                                    metrics['eliminate']['num_total'] += 1
                                    if edit_distance_eliminate == 0:
                                        metrics['eliminate']['num_correct'] += 1
                            
                            if new_size and orig_size and match:#edit_distance_eliminate == 0:
                                s = '<li class="f-correct f-all" style="text-align:left;">\n'
                            else:
                                s = '<li class="f-incorrect f-all" style="text-align:left;">\n'
                            if not orig_size:
                                orig_size = (600,60)
                            if not pre_size:
                                pre_size = (600,60)
                            if not new_size:
                                new_size = (600,60)
                            s += '<div style="height:%dpx; width:%dpx; background-color:white; position:relative;"><img src=%s style="position:absolute; top:%dpx; left:%dpx;"/></div><br/>\n'%(orig_size[1]+PAD_TOP+PAD_BOTTOM,orig_size[0]+PAD_LEFT+PAD_RIGHT,img_rel_orig_path, PAD_TOP, PAD_LEFT)
                            s += 'orig: <br/> %s<br/>\n'%(labels[img_path])
                            s += '<div style="height:%dpx; width:%dpx; background-color:white; position:relative;"><img src=%s style="position:absolute; top:%dpx; left:%dpx;"/></div><br/>\n'%(pre_size[1]+PAD_TOP+PAD_BOTTOM,pre_size[0]+PAD_LEFT+PAD_RIGHT,img_rel_path, PAD_TOP, PAD_LEFT)
                            s += 'gold: <br/> %s (%s)<br/>\n'%(label_gold, score_gold)
                            s += '<div style="height:%dpx; width:%dpx; background-color:white; position:relative;"><img src=%s style="position:absolute; top:%dpx; left:%dpx;"/></div><br/>\n'%(new_size[1]+PAD_TOP+PAD_BOTTOM,new_size[0]+PAD_LEFT+PAD_RIGHT,img_rel_pred_path, PAD_TOP, PAD_LEFT)
                            s += 'predicted: <br/> %s (%s)<br/>\n'%(label_pred, score_pred)
                            if os.path.exists(img_diff_path):
                                im = Image.open(img_diff_path)
                                diff_size = im.size
                                s += '<div style="height:%dpx; width:%dpx; background-color:white; position:relative;"><img src=%s style="position:absolute; top:%dpx; left:%dpx;"/></div><br/>\n'%(diff_size[1]+PAD_TOP+PAD_BOTTOM,diff_size[0]+PAD_LEFT+PAD_RIGHT,img_rel_diff_path, PAD_TOP, PAD_LEFT)
                            s += 'match: %s, edit dist (eliminate): %d / %d; edit dist: %d / %d<br/>\n'%(match, edit_distance_eliminate, ref_eliminate, edit_distance, ref)
                            s += '</li>\n'
                            s += '\n'
                        fout.write(s)


                    for line in ftail:
                        fout.write(line)
    fpred.close()
    fgold.close()
    metric = subprocess.check_output('perl src/evaluation/multi-bleu.perl %s < %s'%(os.path.join(output_dir, 'gold.txt'), os.path.join(output_dir, 'pred.txt')), shell=True)
    print (metric)
    metrics_template = r"""
        <table style="width:100%%">
             <tr>
                 <th></th>
                 <th>#Total</th>
                 <th>Accuracy</th>
                 <th>Accuracy (edit dist)</th>
             </tr>
             <tr>
                 <th>Full</th>
                 <td>%d</td>
                 <td>%f</td>
                 <td>%f</td>
             </tr>
             <tr>
                 <th>Eliminate Spaces</th>
                 <td>%d</td>
                 <td>%f</td>
                 <td>%f</td>
             </tr>
        </table> 
        %s<br/>
    """
    print (metrics)
    with open(html_path, 'r') as fin:
        with open(html_path+'.tmp', 'w') as fout:
            for line in fin:
                if 'MARK: REPLACE' in line:
                    fout.write(metrics_template % (metrics['full']['num_total'], float(metrics['full']['num_correct'])/metrics['full']['num_total'], float(metrics['full']['correct_length'])/metrics['full']['total_length'],
                        metrics['eliminate']['num_total'], float(metrics['eliminate']['num_correct'])/metrics['eliminate']['num_total'], float(metrics['eliminate']['correct_length'])/metrics['eliminate']['total_length'],
                        metric))
                else:
                    fout.write(line)
    shutil.move(html_path+'.tmp', html_path)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
