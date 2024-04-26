`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/04/24 15:34:31
// Design Name: 
// Module Name: Debouncer
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module Debouncer(
        input clk,
        input btn,
        
        output reg btn_state,
        output btn_down,
        output btn_up
    );
    
reg btn_sync_0;
reg btn_sync_1;

reg [15:0] btn_cnt;

wire btn_idle = (btn_state == btn_sync_1);
wire btn_cnt_max = &btn_cnt;

always@(posedge clk) begin
    btn_sync_0 <= ~btn;
    btn_sync_1 <= btn_sync_0;
    
    if (btn_idle)
        btn_cnt <= 0;
    else begin
        btn_cnt <= btn_cnt + 16'd1;
        if (btn_cnt_max) btn_state <= ~btn_state;
    end
end

assign btn_down = ~btn_idle & btn_cnt_max & ~btn_state;
assign btn_up = ~btn_idle & btn_cnt_max & btn_state;

endmodule
