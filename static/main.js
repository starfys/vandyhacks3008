async function main() {
    // Initialize canvas
    const render_canvas = document.getElementById("render");
    const render_ctx = render_canvas.getContext("2d");
    // Initialize the video element
    const video = document.getElementById("video");
    // Open the stream from the webcam
    const stream = await navigator
        .mediaDevices
        .getUserMedia({ video: true });
    // Set the video element to stream the stream from the webcam
    video.srcObject = stream;
    // Play the stream
    await video.play();
    // Create the canvas used to save image data
    const source_canvas = document.createElement("canvas");
    source_canvas.width = video.offsetWidth;
    source_canvas.height = video.offsetHeight;
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
  // Send a request
  const response = await fetch("/transform", {
    method: "POST",
    body: image_blob 
  });
  // Get the response
  const modified_image = await response.blob();
  let image = new Image();
  image.onload = function() {
    render_ctx.drawImage(
      image,
      0, 0,
      video.offsetWidth, video.offsetHeight
    );
  }
  image.src = URL.createObjectURL(modified_image);
  // Wait for the browser to be ready to present another animation frame.
  requestAnimationFrame(function() {
    update(video, source_canvas, render_ctx);
  }); 
}
(async () => {await main()})()
