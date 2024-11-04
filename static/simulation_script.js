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
    // 取得選中的 severity 和 cb_type 的值，並轉換成中文描述
    const severityValue = document.querySelector('input[name="severity"]:checked').value;
    const cbTypeValue = document.querySelector('input[name="cb_type"]:checked').value;
    const severityText = severityValue === "0" ? "輕微" : severityValue === "1" ? "中度" : "嚴重";
    const cbTypeText = cbTypeValue === "0" ? "紅色色覺障礙" : cbTypeValue === "1" ? "綠色色覺障礙" : "藍色色覺障礙";

    // 使模擬後的圖片可下載
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

        // 準備進行邊緣消失比例計算
        const edgeLossFormData = new FormData();
        edgeLossFormData.append('original_image', document.getElementById('upload-img').files[0]);
        // 將模擬圖轉成 Blob 格式後傳送到後端
        fetch(simulatedImage)
            .then(res => res.blob())
            .then(blob => {
                edgeLossFormData.append('simulated_image', blob, 'simulated.png');
                
                // 發送請求到 '/simulation-result'
                return fetch('/simulation-result', {
                    method: 'POST',
                    body: edgeLossFormData
                });
            })
            .then(response => response.json())
            .then(data => {
                // 將計算結果顯示在指定區域
                document.getElementById('simulation-result').innerHTML = `
                    <h4>使用梯度計算方式,  <strong style="font-size: 1.2em;">${severityText}</strong> 的 <strong style="font-size: 1.2em;">${cbTypeText}</strong> 模擬前後的邊緣消失比例</h4>
                    整體邊緣的消失比例為 : <strong style="font-size: 1.5em;">${data.edge_loss_ratio_ori.toFixed(2)}%</strong>
                    <br>分割為4x4區塊後, 邊緣消失比例最大的區塊比例為 : <strong style="font-size: 1.5em;">${data.max_value.toFixed(2)}%</strong>`;
            })
            .catch(error => console.error('Error calculating edge loss:', error));
    })
    .catch(error => console.error('Error:', error));
});
