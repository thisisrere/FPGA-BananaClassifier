// banana_cnn.v
// Input : citra RGB 32x32, 8-bit, urutan img[(c*32 + y)*32 + x]
// Proses: Conv 3x3 x4 -> ReLU -> requantisasi (acc*M)>>>SH, clip 0..255
//         -> MaxPool 2x2 -> FC 900->3
// Output: pred (0=busuk, 1=matang, 2=mentah) dan skor int

module banana_cnn #(
    parameter SH = 16
)(
    input  wire              clk,
    input  wire              rst,
    input  wire              start,
    output reg               done,
    output reg  [1:0]        pred,
    output reg signed [31:0] score0,
    output reg signed [31:0] score1,
    output reg signed [31:0] score2,
    input  wire              img_we,
    input  wire [11:0]       img_addr,
    input  wire [7:0]        img_din
);

    // Dari params.txt (M=232, SH=16). Jika params.txt berubah, ubah nilai ini.
    localparam signed [63:0] M = 232;

    // ---------- Memori ----------
    reg [7:0]         img [0:3071];
    reg signed [7:0]  wc  [0:107];    // 4*3*3*3
    reg signed [31:0] bc  [0:3];
    reg signed [7:0]  wf  [0:2699];   // 3*900
    reg signed [31:0] bf  [0:2];

    initial begin
        $readmemh("w_conv.mem", wc);
        $readmemh("b_conv.mem", bc);
        $readmemh("w_fc.mem",   wf);
        $readmemh("b_fc.mem",   bf);
        $readmemh("test_img.mem", img);
    end

    always @(posedge clk)
        if (img_we) img[img_addr] <= img_din;

    // ---------- Buffer hasil pooling: 4 x 15 x 15 ----------
    reg signed [31:0] pool [0:899];

    // ---------- FSM ----------
    localparam S_IDLE = 3'd0, S_CONV = 3'd1, S_FC = 3'd2, S_OUT = 3'd3, S_DONE = 3'd4;
    reg [2:0] st;

    integer f, pr, pc, dy, dx, r, c, ch, i, j, k;
    reg signed [31:0] acc, v, pmax, s0, s1, s2;
    reg signed [63:0] mul;

    always @(posedge clk) begin
        if (rst) begin
            st   <= S_IDLE;
            done <= 1'b0;
            pred <= 2'd0;
        end else begin
            case (st)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        f = 0; pr = 0; pc = 0;
                        st <= S_CONV;
                    end
                end

                // Satu siklus = satu posisi pooled (f, pr, pc)
                S_CONV: begin
                    pmax = 0;
                    for (dy = 0; dy < 2; dy = dy + 1)
                        for (dx = 0; dx < 2; dx = dx + 1) begin
                            r = pr*2 + dy;
                            c = pc*2 + dx;
                            acc = bc[f];
                            for (ch = 0; ch < 3; ch = ch + 1)
                                for (i = 0; i < 3; i = i + 1)
                                    for (j = 0; j < 3; j = j + 1)
                                        acc = acc + $signed({1'b0, img[(ch*32 + r + i)*32 + (c + j)]})
                                                  * wc[f*27 + ch*9 + i*3 + j];
                            if (acc < 0) acc = 0;                    // ReLU
                            mul = ($signed(acc) * M) >>> SH;         // requantisasi
                            if (mul > 255) mul = 255;                // clip 0..255
                            v = mul[31:0];
                            if (v > pmax) pmax = v;                  // MaxPool 2x2
                        end

                    pool[f*225 + pr*15 + pc] = pmax;

                    if (pc == 14) begin
                        pc = 0;
                        if (pr == 14) begin
                            pr = 0;
                            if (f == 3) begin
                                f = 0;
                                st <= S_FC;
                            end else f = f + 1;
                        end else pr = pr + 1;
                    end else pc = pc + 1;
                end

                // FC: 3 skor, dihitung dalam satu siklus
                S_FC: begin
                    s0 = bf[0]; s1 = bf[1]; s2 = bf[2];
                    for (k = 0; k < 900; k = k + 1) begin
                        s0 = s0 + pool[k] * wf[0*900 + k];
                        s1 = s1 + pool[k] * wf[1*900 + k];
                        s2 = s2 + pool[k] * wf[2*900 + k];
                    end
                    st <= S_OUT;
                end

                S_OUT: begin
                    score0 <= s0;
                    score1 <= s1;
                    score2 <= s2;
                    // argmax, seri diambil indeks kecil (sama dengan np.argmax)
                    pred   <= (s0 >= s1 && s0 >= s2) ? 2'd0 :
                              (s1 >= s2)             ? 2'd1 : 2'd2;
                    done   <= 1'b1;
                    st     <= S_DONE;
                end

                S_DONE: begin
                    if (!start) st <= S_IDLE;
                end

                default: st <= S_IDLE;
            endcase
        end
    end

endmodule