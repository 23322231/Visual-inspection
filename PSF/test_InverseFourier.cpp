#include <iostream>
#include <fftw3.h>

int main() {
    const int N = 4; // 假設為4x4的二維陣列
    fftw_complex in[N][N], out[N][N];
    fftw_plan p;

    // 初始化輸入數據，這裡僅示例全零
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            in[i][j][0] = 0, in[i][j][1] = 0;

    // 創建逆傅里葉變換計劃
    p = fftw_plan_dft_2d(N, N, *in, *out, FFTW_BACKWARD, FFTW_ESTIMATE);

    // 執行逆傅里葉變換
    fftw_execute(p);

    // 輸出結果
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            std::cout << out[i][j][0] / (N * N) << " + " << out[i][j][1] / (N * N) << "i ";
        std::cout << std::endl;
    }

    // 清理資源
    fftw_destroy_plan(p);
    fftw_cleanup();

    return 0;
}
