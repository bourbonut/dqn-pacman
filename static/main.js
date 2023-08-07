const svg_ids = [
  "losses_raw",
  "rewards_mean",
  "qvalues_mean",
  "rewards_raw",
  "rewards_total",
  "qvalues_total",
];

const xlabels = {
  "losses_raw": "Steps",
  "rewards_mean": "Episodes",
  "qvalues_mean": "Episodes",
  "rewards_raw": "Steps",
  "rewards_total": "Episodes",
  "qvalues_total": "Episodes",
}

const ylabels = {
  "losses_raw": "Losses",
  "rewards_mean": "Reward Mean",
  "qvalues_mean": "Q Value Mean",
  "rewards_raw": "Rewards",
  "rewards_total": "Total Of Rewards",
  "qvalues_total": "Total Of Q Values",
}

const colors = {
  "losses_raw": "#FFA233",
  "rewards_mean": "#007ea7",
  "qvalues_mean": "yellow",
  "rewards_raw": "#007ea7",
  "rewards_total": "#007ea7",
  "qvalues_total": "yellow",
}

var data = Object.fromEntries(
  svg_ids.map(svg_id => [svg_id, {"x":[], "y":[]}])
);

const margin = {top: 30, right: 30, bottom: 50, left: 80},
  width = (window.screen.width - 350) / 4.8;
  height = width * 2 / 3

function zip(arrays) {
    var keys = Object.keys(arrays);
    var arrays = keys.map(function(key) { return arrays[key] })
    return Array.apply(null, Array(arrays[0].length)).map(function(_, i) {
        return arrays.map(function(array) { return array[i] })
    });
}

function lineChart(svg_id, data){
  var svg = d3.select("#" + svg_id)
    .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
    .append("g")
      .attr("transform",
            "translate(" + margin.left + "," + margin.top + ")");

  var keys = Object.keys(data);
  var data_ready = zip(data).map(
    e => Object.fromEntries(
      e.map((d, i) => [keys[i], d])
    )
  );

  var xmax = data.x.reduce((a, b) => Math.max(a, b), 0);
  var x = d3.scaleLinear().domain([0, xmax]).range([0, width]).nice();
  svg.append("g")
    .attr("class", "xaxis")
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(x).ticks(10, "s"));
  svg.append("text")
      .attr("class", "xlabel")
      .attr("text-anchor", "middle")
      .attr("x", width / 2)
      .attr("y", height + 35)
      .attr("fill", "white")
      .attr("stroke", "none")
      .text(xlabels[svg_id]);

  // Add y axis and y label
  var ymin = data.y.reduce((a, b) => Math.min(a, b), 0);
  var ymax = data.y.reduce((a, b) => Math.max(a, b), 0);
  var y = d3.scaleLinear().domain([ymin, ymax]).range([ height, 0 ]);
  svg.append("g")
    .attr("class", "yaxis")
    .call(d3.axisLeft(y));

  svg.append("text")
      .attr("class", "ylabel")
      .attr("text-anchor", "middle")
      .attr("x", -height / 2)
      .attr("y", -2 * margin.left / 3)
      .attr("transform", "rotate(-90)")
      .attr("fill", "white")
      .attr("stroke", "none")
      .text(ylabels[svg_id]);

  svg.selectAll("path.domain").attr("stroke", "white");
  svg.selectAll("g.tick").selectAll("line").attr("stroke", "white");
  svg.selectAll("g.tick").selectAll("text").attr("fill", "white").attr("stroke", "none");

  var line = svg
    .append('g')
    .append("path")
      .datum(data_ready)
      .attr("d", d3.line()
        .x(function(d) { return x(d.x) })
        .y(function(d) { return y(d.y) })
      )
      .attr("stroke", colors[svg_id] )
      .style("stroke-width", 2)
      .style("fill", "none")

  return {svg: svg, line: line, x: x, y: y};
}

function update(svg, line, x, y, data) {
  // Create new data with the selection
  var keys = Object.keys(data);
  var data_ready = zip(data).map(
    e => Object.fromEntries(
      e.map((d, i) => [keys[i], d])
    )
  );

  // Update domains and axis
  var xmax = data.xmax;
  var ymin = data.ymin;
  var ymax = data.ymax;

  x.domain([0, xmax]).nice();
  y.domain([ymin, ymax]);
  svg.selectAll("g.xaxis").
      // transition().
      // duration(50).
      // ease(d3.easePoly).
      call(d3.axisBottom(x).ticks(10, "s"));
  svg.selectAll("g.yaxis").
      // transition().
      // duration(50).
      // ease(d3.easePoly).
      call(d3.axisLeft(y));

  // Give these new data to update line
  line
    .datum(data_ready)
    // .transition()
    // .duration(2)
    // .ease(d3.easePoly)
    .attr("d", d3.line()
      .x(function(d) { return x(d.x) })
      .y(function(d) { return y(d.y) })
    )
  svg.selectAll("g.tick").selectAll("line").attr("stroke", "white");
  svg.selectAll("g.tick").selectAll("text").attr("fill", "white").attr("stroke", "none");
}

var elements = Object.fromEntries(
  svg_ids.map(
    svg_id => [svg_id, lineChart(svg_id, data[svg_id])]
  )
);

let socket = new WebSocket("ws://localhost:5000/ws"); // 8765

socket.onopen = function(e) {
  console.log("[open] Connection established");
};

socket.onmessage = function(event) {
  var received_data = JSON.parse(event.data);
  document.getElementById("pacman").src = "data:image/png;base64," + received_data.image;
  for (svg_id of svg_ids){
    var element = elements[svg_id];
    update(element.svg, element.line, element.x, element.y, received_data[svg_id]);
  }
};

socket.onclose = function(event) {
  if (event.wasClean) {
    console.log(`[close] Connection closed cleanly, code=${event.code} reason=${event.reason}`);
  } else {
    console.log('[close] Connection died');
  }
};

socket.onerror = function(error) {
  console.log(`[error]`);
};
