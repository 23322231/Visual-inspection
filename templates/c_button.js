function confirm(){ 
    const socket = io();
    

        socket.on('button_press', (msg) => {
            console.log('Button Pressed on the server:', msg);
        });
    
        function sendButtonPress() {
            socket.emit('button_press', 'Button Pressed!');
        }
}