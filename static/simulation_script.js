// 正式版
document.getElementById('upload-img').addEventListener('change', function(event) {
    const file = event.target.files[0];
    const reader = new FileReader();

    reader.onload = function(e) {
        document.getElementById('before-img').src = e.target.result;
    };

    if (file) {
        reader.readAsDataURL(file);
    }
});

document.querySelector('form').addEventListener('submit', function(event) {
    event.preventDefault();

    const formData = new FormData(this);

    fetch('/simulate', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const simulatedImage = data.image_data;
        document.getElementById('after-img').src = simulatedImage;

        const downloadBtn = document.getElementById('download-btn');
        downloadBtn.addEventListener('click', function() {
            const a = document.createElement('a');
            a.href = simulatedImage;

            const fileExt = simulatedImage.split(';')[0].split('/')[1];
            a.download = `simulated.${fileExt}`;

            a.style.display = 'none';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        });
    })
    .catch(error => console.error('Error:', error));
});
