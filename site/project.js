require(["d3"], init);

var margin = {top: -120, right: 10, bottom: 0, left: 10},
    width = 1200 - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom;

var rows = 4;

function trans(x, y) {
    return "translate("+ x + "," + y + ")";
}

var all_data
var results_data

function init(d3) {

    var svg = d3.select("#align").style("text-align", "center").append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    function make_text(data, id, img_width, img_height, focus) {
    if (img_height > 0) {
        scale =  width / img_width;
        d3.select("svg").attr("height", img_height*scale + margin.top + margin.bottom+175)
    }
        var cols = data[0].words.length;
        var xScale = d3.scale.linear()
            .domain([0, cols])
            .range([0, width]);

        var yScale = d3.scale.linear()
            .domain([0, rows])
            .range([0, height]);
        var xMapScale, yMapScale;

        var base = svg.selectAll("g")
            .data([data[0]], function(d) { return d.img;})

        var base_ent = base.enter()
            .append("g");

        base.exit().remove();

        var g = base.selectAll("g")
            .data(function(d){return d.words;});

        var g2 = g.enter()
            .append("g");
        g.exit().remove();

        scale =  width / img_width;
        imageStartX = 0;
        imageStartY = 175;
        diffY = 8;
        diffX = 8;
        var yImScale = d3.scale.linear()
            .domain([0, img_height/8])
            .range([imageStartY, imageStartY + img_height *scale]);
        var xImScale = d3.scale.linear()
            .domain([0, img_width/8])
            .range([imageStartX, imageStartX + img_width * scale]);

        g2
            .append("text")
            .attr("dx", function(d, i) { return xScale(i); } )
            .attr("dy", function(d, i) { return yScale(1); })
            .on("mouseover", function(d) {make_text(data, id, img_width, img_height, d.scores);} )
            .text(function(d) {return d.word.substr(0, 6);} );

        g2.append("line")
            .style("stroke-dasharray", "5,5")
            .style("stroke", function (d) { return "black";

                                          } )
            .style("display", function(d) {
                if (d.word == "\\mathrm" || d.word == "\\bf" || d.word == "{" || d.word == "}" || d.word == "_" || d.word == "^") {
                    return "none";
                } else {
                    return "block";
                }
            }
                  )
            .attr("x1", function(d, i) { return xScale(i); })
            .attr("y1", function(d, i) { return yScale(1); })
            .attr("x2", function(d, i) { return xImScale(d.exp_col -0.5); })
            .attr("y2", function(d, i) { return yImScale(d.exp_row - 0.5); });

        var b = base.selectAll("rect.a")
            .data(focus, function(d) { return d.row + "" + d.col;} );

        b.enter()
            .append("rect")
            .attr("width", xImScale(1) - xImScale(0))
            .attr("height", yImScale(1) - yImScale(0))
            .attr("x", function(d) { return xImScale(d.col-1);})
            .attr("y", function(d) { return yImScale(d.row-1);})
            .attr("class", "a");

        b.exit().remove();
        b.style("fill", "red")

            .style("opacity", function(d) {return d.score/2;} );

        b.exit().remove();

        b = base.selectAll("rect.b")
            .data(focus, function(d) { return d.row + "" + d.col;} );

        b.enter()
            .append("rect")
            .attr("width", xImScale(1) - xImScale(0))
            .attr("height", yImScale(1) - yImScale(0))
            .attr("x", function(d) { return xImScale(d.col-1);})
            .attr("y", function(d) { return yImScale(d.row-1);})
            .attr("class", "b");

        b.exit().remove();
        b.style("fill", "blue")

            .style("opacity", function(d) {return d.score_coarse/4;} );

        b.exit().remove();

        for (var i = 0; i <= img_height/8; i++) {
            base_ent.insert("line", ":first-child")
                .attr("x1", xImScale(0))
                .attr("y1", yImScale(i))
                .attr("x2", xImScale(img_width/8))
                .attr("y2", yImScale(i))
                .style("stroke", "black")
                .style("opacity", "0.3")
        }

        for (var i = 0; i <= img_width/8; i++) {
            base_ent.insert("line", ":first-child")
                .attr("y1", yImScale(0))
                .attr("x1", xImScale(i))
                .attr("y2", yImScale(img_height/8))
                .attr("x2", xImScale(i))
                .style("stroke", "black")
                .style("opacity", "0.3")
        }

        base_ent.insert("image", ":first-child")
            .attr("xlink:href", function(d){return "img/" + d.img;})
            .attr("x", xImScale(0))
            .attr("y", yImScale(0))
            .attr("width", img_width *scale )
            .attr("height", img_height * scale);

    }

    function show(im_num) {
        var img = new Image();
        img.onload = function() {
            make_text([all_data[im_num]], im_num, this.width, this.height, []);
        }
        img.src = "img/" + all_data[im_num].img;

    }

    d3.json("vis_out.json", function(error, data) {
        if (error || !data || !data.length) {
            d3.select("#alignment-status").text("The alignment examples could not be loaded. Please reload this page to try again.");
            return;
        }
        d3.select("#alignment-status").remove();
        all_data = data

        show(0);
        var cur = 0;
        d3.select("#align").insert("button", ":first-child").attr("id","preva").attr("type", "button").attr("href", "#").attr("class", "btn btn-outline-primary").style("border-width", "1px").style("background-color","white").style("border-color","#DCDCDC").style("hover","backg").text("prev").on("click", function() {
            cur--;
            if (cur==0)  {
                d3.select(this).attr('disabled', true);
            }
            if (cur<(all_data.length-1))  {
                d3.select('#nexta').attr('disabled', null);
                d3.select("#nexta").style('background-color', "white");
            }
            show(cur);
	    this.blur();
         } ).on("mouseover", function(){d3.select(this).style("background-color","black")}).on("mouseout", function(){d3.select(this).style("background-color", "white")}).on("focus", function(){if (document.activeElement != document.body) document.activeElement.blur()});
        d3.select("#align").insert("button", ":first-child").attr("id","nexta").attr("type", "button").attr("href", "#").attr("class", "btn btn-outline-primary").style("border-width", "1px").style("background-color","white").style("border-color","#DCDCDC").style("hover","backg").text("next").on("click", function() {
            cur++;
            if (cur==(all_data.length-1))  {
                d3.select(this).attr('disabled', true);
            }
            if (cur>0)  {
                d3.select("#preva").attr('disabled', null);
                d3.select("#preva").style('background-color', "white");
            }
            show(cur);
            } ).on("mouseover", function(){d3.select(this).style("background-color","black")}).on("mouseout", function(){d3.select(this).style("background-color", "white")}).on("focus", function(){if (document.activeElement != document.body) document.activeElement.blur()});
        if (cur==0)  {
            d3.select("#preva").attr('disabled', true);
        }
        if (cur==(all_data.length-1))  {
            d3.select("#nexta").attr('disabled', true);
        }
    });

    d3.json("results.json", function(error, data) {
        if (error || !data || !data.length) {
            d3.select("#results-status").text("The qualitative results could not be loaded. Please reload this page to try again.");
            return;
        }
        d3.select("#results-status").remove();
        var cur_result = 0;
        results_data = data;
        showResult(0);
        d3.select("#results").insert("button", ":first-child").attr("id","prevresults").attr("type", "button").attr("href", "#").attr("class", "btn btn-outline-primary").style("border-width", "1px").style("background-color","white").style("border-color","#DCDCDC").style("hover","backg").text("prev").on("click", function() {
        cur_result--;
        if (cur_result==0)  {
            d3.select(this).attr('disabled', true);
        }
        if (cur_result<(results_data.length-1))  {
            d3.select('#nextresults').attr('disabled', null);
            d3.select("#nextresults").style('background-color', "white");
        }
        showResult(cur_result);
        } ).on("mouseover", function(){d3.select(this).style("background-color","black")}).on("mouseout", function(){d3.select(this).style("background-color", "white")}).on("focus", function(){if (document.activeElement != document.body) document.activeElement.blur()});
        d3.select("#results").insert("button", ":first-child").attr("id","nextresults").attr("type", "button").attr("href", "#").attr("class", "btn btn-outline-primary").style("border-width", "1px").style("background-color","white").style("border-color","#DCDCDC").style("hover","backg").text("next").on("click", function() {
            cur_result++;
            if (cur_result==(results_data.length-1))  {
                d3.select(this).attr('disabled', true);
            }
            if (cur_result>0)  {
                d3.select("#prevresults").attr('disabled', null);
                d3.select("#prevresults").style('background-color', "white");
            }
            showResult(cur_result);
        } ).on("mouseover", function(){d3.select(this).style("background-color","black")}).on("mouseout", function(){d3.select(this).style("background-color", "white")}).on("focus", function(){if (document.activeElement != document.body) document.activeElement.blur()});
        if (cur_result==0)  {
            d3.select("#prevresults").attr('disabled', true);
        }
        if (cur_result==(results_data.length-1))  {
            d3.select("#nextresults").attr('disabled', true);
        }
    });
    function showResult(im_num) {
        var orig_width = parseInt(results_data[im_num].orig[0], 10);
        var orig_height = parseInt(results_data[im_num].orig[1], 10);
        var line = results_data[im_num].orig[3]
        if (results_data[im_num].hasOwnProperty('eval')) {
            if (results_data[im_num].eval.includes('match: False')) {
                str = '<font color="red">'+results_data[im_num].eval+'</font>';
            } else {
                str = '<font color="green">'+'match: True'+'</font><br/>';
            }
        }
        str = str+line;
        str = str+'<div id="orig" style="height:'+(orig_height+16).toString()+'px; width:'+(orig_width+16).toString()+'px; background-color:white; position:relative;"><img id="orig-img" src="results/website/images/'+results_data[im_num].orig[2]+'" style="position:absolute; top:8px; left:8px;"/></div><br/>';
        if (results_data[im_num].hasOwnProperty('pred')) {
            str = str+results_data[im_num].pred[3]
            var pred_width = parseInt(results_data[im_num].pred[0], 10);
            var pred_height = parseInt(results_data[im_num].pred[1], 10);
            str = str+'<div id="pred" style="height:'+(pred_height+16).toString()+'px; width:'+(pred_width+16).toString()+'px; background-color:white; position:relative;"><img id="pred-img" src="results/website/images/'+results_data[im_num].pred[2]+'" style="position:absolute; top:8px; left:8px;"/></div><br/>';
        }
        if (results_data[im_num].hasOwnProperty('diff')) {
            if (results_data[im_num].eval.includes('match: False')) {
                var diff_width = parseInt(results_data[im_num].diff[0], 10);
                var diff_height = parseInt(results_data[im_num].diff[1], 10);
                str = str+'diff image:<br/>';
                str = str+'<div id="pred" style="height:'+(diff_height).toString()+'px; width:'+(diff_width).toString()+'px; background-color:white; position:relative;"><img id="diff-img" src="results/website/images/'+results_data[im_num].diff[2]+'" style="position:absolute; top:0px; left:0px;"/></div><br/>';
            }
        }
        d3.select('#results-img').html(str);
    }
}
