// Canvas DOM 元素 
const canvas = document.getElementById('canvas')
const ctx = canvas.getContext('2d')

//起始點座標
let x1= 0
let y1= 0

// 終點座標
let x2= 0
let y2= 0

// 宣告一個 hasTouchEvent 變數，來檢查是否有 touch 的事件存在
const hasTouchEvent = 'touchstart' in window ? true : false

// 透過上方的 hasTouchEvent 來決定要監聽的是 mouse 還是 touch 的事件
//const downEvent = hasTouchEvent ? 'touchstart' : 'mousedown'
//const moveEvent = hasTouchEvent ? 'touchmove' : 'mousemove'
//const upEvent = hasTouchEvent ? 'touchend' : 'mouseup'
const downEvent ='touchstart'
const moveEvent ='touchmove'
const upEvent ='touchend'
// 宣告 isMouseActive 為滑鼠點擊的狀態，因為我們需要滑鼠在 mousedown 的狀態時，才會監聽 mousemove 的狀態

let isMouseActive = false

canvas.addEventListener(downEvent, function(e){
  isMouseActive = true
})

canvas.addEventListener(downEvent, function(e){
  isMouseActive = true  
  x1 = e.clientX
  y1 = e.clientY

  ctx.lineWidth = 5
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
})
// 儲存線條軌跡的座標陣列
var lineCoordinates = [];
canvas.addEventListener(moveEvent, function(e){
      if(!isMouseActive){
        return
      }
      // 取得終點座標
      x2 = e.clientX
      y2 = e.clientY
      
      // 開始繪圖
      ctx.beginPath()
      ctx.moveTo(x1, y1)
      ctx.lineTo(x2, y2)
      ctx.stroke()

      // 將座標紀錄到軌跡陣列
      lineCoordinates.push([x2,y2]);

      // 更新起始點座標
      x1 = x2
      y1 = y2
      
})

//結束繪圖
canvas.addEventListener(upEvent, function(e){
  isMouseActive = false
  // 輸出軌跡座標
  
  //document.write(lineCoordinates);
  // 清空軌跡座標陣列，以便下次繪製
  lineCoordinates = [];
})

