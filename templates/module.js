// Create an AudioContext
const audioContext = new AudioContext();

// Create an audio input
const audioInput = audioContext.createMediaStreamSource(navigator.mediaDevices.getUserMedia({audio: true}));

// Create a recorder
const recorder = new Recorder(audioContext);

// Handle the "record" button click
$(document).on('click', '#btn-record', () => {
    // Start recording
    recorder.start();
});

// Handle the "stop" button click
$(document).on('click', '#btn-stop', () => {
    // Stop recording
    recorder.stop();

    // Get the recorded audio
    const audioBuffer = recorder.getBuffer();

    // Send the recorded audio to the API
    $.ajax({
        url: '/predict',
        method: 'POST',
        data: {
            audio: audioBuffer
        }
    }).done((response) => {
        $('#prediction').text(response);
    });
});
