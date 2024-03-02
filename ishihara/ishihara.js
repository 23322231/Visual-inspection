'use strict'; // 變數要宣告後才可以使用
//用來計算解析度
var PIXEL_RATIO = window.devicePixelRatio || 1;

document.addEventListener('DOMContentLoaded',function(){
    var canvas = document.getElementById('canvas');
    //在 HTML 中建立畫畫的 object
    var ctx = canvas.getContext("2d");

    //設定畫框大小
    ctx.canvas.style.width = window.innerWidth + 'px' //因為要轉成 css 的的格式，所以要加 'px'
    ctx.canvas.style.height = window.innerHeight + 'px'

    //設定畫紙大小(設定為螢幕的真實解析度)
    ctx.canvas.width = window.innerWidth * PIXEL_RATIO;
    ctx.canvas.height = window.innerHeight * PIXEL_RATIO;

    ctx.fillStyle = "black";
    ctx.fillRect(0,0,canvas.width,canvas.height);
});