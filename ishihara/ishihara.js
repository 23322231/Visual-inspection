'use strict'; // 變數要宣告後才可以使用
//用來計算解析度
var PIXEL_RATIO = window.devicePixelRatio || 1;

document.addEventListener('DOMContentLoaded',function(){
    var toolbar = document.getElementById('toolbar');
    var canvas = document.getElementById('canvas');
    //在 HTML 中建立畫畫的 object
    var ctx = canvas.getContext("2d");

    var toolbarOffset = canvas.offsetLeft;
    var topOffset = canvas.offsetTop;

    //設定畫框大小
    ctx.canvas.style.width = window.innerWidth + 'px' //因為要轉成 css 的的格式，所以要加 'px'
    ctx.canvas.style.height = window.innerHeight + 'px'

    //設定畫紙大小(設定為螢幕的真實解析度)
    ctx.canvas.width = window.innerWidth * PIXEL_RATIO - toolbarOffset; //剪掉旁邊 toolbar 的部分
    ctx.canvas.height = window.innerHeight * PIXEL_RATIO - topOffset;

    // ctx.fillStyle = "white";
    // ctx.fillRect(0,0,canvas.width,canvas.height);

    let isPainting = false;
    let lineWidth = 5;
    let startX;
    let startY;

    // toolbar 的事件監聽器
    toolbar.addEventListener('click',e => {
        if(e.target.id === 'clear'){
            ctx.clearRect(0,0,ctx.canvas.width,ctx.canvas.height);
        }
    });
    toolbar.addEventListener('change',e =>{
        if(e.target.id === 'lineWidth'){
            lineWidth = e.target.value;
        }
    });

    //滑鼠移動時會呼叫的畫筆(？
    const draw = (e) =>{
        if(!isPainting){
            return;
        }
        ctx.lineWidth=lineWidth;
        ctx.lineCap='round';

        ctx.lineTo(e.clientX*PIXEL_RATIO-toolbarOffset,e.clientY*PIXEL_RATIO);
        ctx.stroke();
    }

    // 畫畫的地方的事件監聽器
    canvas.addEventListener('mousedown',e =>{
        isPainting = true;
        startX = e.clientX;
        startY = e.clientY;
    });
    canvas.addEventListener('mouseup',e =>{
        isPainting = false;
        ctx.stroke();
        ctx.beginPath();
    });
    canvas.addEventListener('mousemove',draw);
});