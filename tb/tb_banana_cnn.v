// tb_banana_cnn.v
`timescale 1ns/1ps
module tb_banana_cnn;
    reg clk = 0, rst = 1, start = 0;
    wire        done;
    wire [1:0]  pred;
    wire signed [31:0] s0, s1, s2;

    banana_cnn dut (
        .clk(clk), .rst(rst), .start(start),
        .done(done), .pred(pred),
        .score0(s0), .score1(s1), .score2(s2),
        .img_we(1'b0), .img_addr(12'd0), .img_din(8'd0)
    );

    always #5 clk = ~clk;

    initial begin
        #20 rst = 0;
        @(posedge clk); start <= 1;
        @(posedge clk); start <= 0;
        wait (done);
        #1;
        $display("RTL  : pred=%0d skor=%0d %0d %0d", pred, s0, s1, s2);
        $finish;
    end

    initial begin
        #20_000_000;
        $display("TIMEOUT");
        $finish;
    end
endmodule