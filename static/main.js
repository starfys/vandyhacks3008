const features = [
  {name: "black_hair", description: "Black Hair", id: 0},
  {name: "blond_hair", description: "Blonde Hair", id: 1},
  {name: "brown_hair", description: "Brown Hair", id: 2},
  {name: "male", description: "Male", change_description: "Change Gender", id: 3},
  {name: "young", description: "Young", id: 4},
];
let target_feature = features[3].id;

let cur_features = [1,0,0,1,1];

async function main() {
    // Initialize canvas
    const render_canvas = document.getElementById("render");
    const render_ctx = render_canvas.getContext("2d");
    // Initialize the video element
    const video = document.getElementById("video");
    // Open the stream from the webcam
    const stream = await navigator
        .mediaDevices
        .getUserMedia({video: { width: 1280, height: 720 }});
    // Set the video element to stream the stream from the webcam
    video.srcObject = stream;
    // Play the stream
    await video.play();
    // Create the canvas used to save image data
    const source_canvas = document.createElement("canvas");
    source_canvas.width = video.offsetWidth;
    source_canvas.height = video.offsetHeight;
	render_canvas.width = video.offsetWidth;
    render_canvas.height = video.offsetHeight;
	// Set up the feature radio buttons
	const features_div = document.getElementById("features");
	
    features_div.classList.add("btn-group");
    features_div.classList.add("btn-group-toggle");
    features_div.setAttribute("data-toggle", "buttons");
	const curfeatures_div = document.getElementById("cur_features");
	//curfeatures_div.classList.add("btn-group");
    curfeatures_div.classList.add("btn-group-toggle");
    curfeatures_div.setAttribute("data-toggle", "buttons");
	features.forEach((feature) => {
	  const label = document.createElement("label");
	  label.for = feature.name;
	  label.innerHTML = feature.description;
	  if(feature.id == 3) {
		label.innerHTML = "Change Gender";
	  }
	  label.classList.add("btn");
	  label.classList.add("btn-lg");
	  label.classList.add("btn-secondary");
	  
	  const r = document.createElement("input");
	  r.type = "radio";
	  r.name = "feature";
	  r.id = feature.name;
	  r.value = feature.id;
	  r.onchange = function() {
		if (this.value) {
			target_feature = this.value;
		}
	  }
	  if(target_feature == feature.id) {
		r.checked = true;
		label.classList.add("active");
	  }
	  label.appendChild(r);
	  features_div.appendChild(label);
	});
	features.forEach((feature) => {
	  const label = document.createElement("label");
	  label.for = feature.name;
	  label.innerHTML = feature.description;
	  label.classList.add("btn");
	  label.classList.add("btn-lg");
	  label.classList.add("btn-secondary");
	  
	  const r = document.createElement("input");
	  r.type = "checkbox";
	  r.name = "feature";
	  r.id = feature.name;
	  r.value = feature.id;
	  r.onchange = function() {
		if (this.checked) {
			cur_features[this.value] = 1;
		}
		else {
			cur_features[this.value] = 0;
		}
		console.log(cur_features);
	  }
	if(cur_features[r.value]) {
		label.classList.add("active");
	}
	  label.appendChild(r);
	  curfeatures_div.appendChild(label);
	});
    //Start rendering the frames to the stream
    update(video, source_canvas, render_ctx);
}

async function update(video, source_canvas, render_ctx) {
  const source_ctx = source_canvas.getContext("2d");
  // Draw the frame from the webcam stream to the canvas 
  source_ctx.drawImage(
      video,
      0, 0,
      video.offsetWidth, video.offsetHeight
  );
  // Copy data out of the canvas
  const image_blob = source_canvas.toDataURL("image/jpeg");
  let c = "";
  for(let i = 0; i < 5; i++) {
	c+=cur_features[i].toString();
  }
  // Send a request
  let blob = null;
  try {
    const response = await fetch("/transform", {
	  method: "POST",
	  headers: { 'Target': target_feature, 'Current': c},
	  body: image_blob 
	});
	// Get the response
	modified_image = await response.blob();
  }
  catch(e) {
	modified_image = new Blob();
  }
  if (modified_image.size != 0) {
	  let image = new Image();
	  image.onload = function() {
		render_ctx.drawImage(
		  image,
		  0, 0,
		  render_ctx.canvas.clientWidth, render_ctx.canvas.clientHeight
		);
	  }
	  image.src = URL.createObjectURL(modified_image);
  }
  else {
	render_ctx.drawImage(
	  video,
	  0, 0,
	  render_ctx.canvas.clientWidth, render_ctx.canvas.clientHeight
	);
  }
  // Wait for the browser to be ready to present another animation frame.
  requestAnimationFrame(function() {
    update(video, source_canvas, render_ctx);
  }); 
}
(async () => {await main()})()
