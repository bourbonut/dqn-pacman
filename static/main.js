const svg_ids = [
  "losses_raw",
  "rewards_mean",
  "qvalues_mean",
  "rewards_raw",
  "rewards_total",
  "qvalues_total",
];

var data = Object.fromEntries(
  svg_ids.map(svg_id => [svg_id, {"x":[], "y":[]}])
);

const margin = {top: 30, right: 250, bottom: 50, left: 70},
  width = 480
  height = 270

function zip(arrays) {
    var keys = Object.keys(arrays);
    var arrays = keys.map(function(key) { return arrays[key] })
    return Array.apply(null, Array(arrays[0].length)).map(function(_, i) {
        return arrays.map(function(array) { return array[i] })
    });
}

function range(array) {
  return [...Array(array.length).keys()]
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
  var xlabel = svg.append("text")
      .attr("class", "xlabel")
      .attr("text-anchor", "middle")
      .attr("x", width / 2)
      .attr("y", height + 35)
      .text("x")

  // Add y axis and y label
  var ymin = data.y.reduce((a, b) => Math.min(a, b), 0);
  var ymax = data.y.reduce((a, b) => Math.max(a, b), 0);
  var y = d3.scaleLinear().domain([ymin, ymax]).range([ height, 0 ]).nice();
  svg.append("g")
    .attr("class", "yaxis")
    .call(d3.axisLeft(y).ticks(10, "s"));

  var ylabel = svg.append("text")
      .attr("class", "ylabel")
      .attr("text-anchor", "middle")
      .attr("x", -height / 2)
      .attr("y", -2 * margin.left / 3)
      .attr("transform", "rotate(-90)")
      .text("y")

  var line = svg
    .append('g')
    .append("path")
      .datum(data_ready)
      .attr("d", d3.line()
        .x(function(d) { return x(d.x) })
        .y(function(d) { return y(d.y) })
      )
      .attr("stroke", "blue" )
      .style("stroke-width", 2)
      .style("fill", "none")

  return {svg: svg, line: line, x: x, y: y};
}

function update(svg, line, x, y, data) {
  // Create new data with the selection
  data.x = range(data.y);
  var keys = Object.keys(data);
  var data_ready = zip(data).map(
    e => Object.fromEntries(
      e.map((d, i) => [keys[i], d])
    )
  );

  // Update domains and axis
  var xmax = data.x.reduce((a, b) => Math.max(a, b), 0);
  var ymin = data.y.reduce((a, b) => Math.min(a, b), 0);
  var ymax = data.y.reduce((a, b) => Math.max(a, b), 0);

  x.domain([0, xmax]).nice();
  y.domain([ymin, ymax]).nice();
  svg.selectAll("g.xaxis").
      transition().
      duration(50).
      ease(d3.easePoly).
      call(d3.axisBottom(x).ticks(10, "s"));
  svg.selectAll("g.yaxis").
      transition().
      duration(50).
      ease(d3.easePoly).
      call(d3.axisLeft(y).ticks(10, "s"));

  // Give these new data to update line
  line
    .datum(data_ready)
    .transition()
    .duration(2)
    .ease(d3.easePoly)
    .attr("d", d3.line()
      .x(function(d) { return x(d.x) }) // .x due to getDataFilter
      .y(function(d) { return y(d.y) })
    )
}

var elements = Object.fromEntries(
  svg_ids.map(
    svg_id => [svg_id, lineChart(svg_id, data[svg_id])]
  )
);

let socket = new WebSocket("ws://localhost:5000/ws"); // 8765

socket.onopen = function(e) {
  console.log("[open] Connection established");
  // console.log("Sending to server");
  // socket.send("My name is John");
};

socket.onmessage = function(event) {
  data = JSON.parse(event.data);
  document.getElementById("pacman").src = "data:image/png;base64," + data.image;
  for (svg_id of svg_ids){
    var element = elements[svg_id];
    update(element.svg, element.line, element.x, element.y, data[svg_id]);
  }
};

socket.onclose = function(event) {
  if (event.wasClean) {
    console.log(`[close] Connection closed cleanly, code=${event.code} reason=${event.reason}`);
  } else {
    // e.g. server process killed or network down
    // event.code is usually 1006 in this case
    console.log('[close] Connection died');
  }
};

socket.onerror = function(error) {
  console.log(`[error]`);
};
