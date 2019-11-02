async function main() {
    // Initialize canvas
    const canvas = document.getElementById('render');
    const ctx = canvas.getContext("2d");
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
    // Render the frame to the stream
    ctx.drawImage(
      video,
      0, 0,
      video.offsetWidth, video.offsetHeight
    );
}
(async () => {await main()})()
