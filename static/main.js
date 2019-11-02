let video = null;
let ctx = null;

async function main() {
    // Initialize canvas
    const canvas = document.getElementById('render');
    ctx = canvas.getContext("2d");
    // Initialize the video element
    video = document.getElementById("video");
    // Open the stream from the webcam
    const stream = await navigator
        .mediaDevices
        .getUserMedia({ video: true });
    // Set the video element to stream the stream from the webcam
    video.srcObject = stream;
    // Play the stream
    await video.play();
    //Start rendering the frames to the stream
      update();
     }



function update(){
  ctx.drawImage(video,0,0,video.offsetWidth, video.offsetHeight);
  // Wait for the browser to be ready to present another animation frame.
  requestAnimationFrame(update); 
}
(async () => {await main()})()
