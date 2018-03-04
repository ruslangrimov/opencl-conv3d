__kernel
//
void vol2col(
    const int n,
    __global const float* data_vol,
    const int vol_offset,
    const int channels,
    const int depth,
    const int height,
    const int width,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int pad_d,
    const int pad_h,
    const int pad_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int depth_col,
    const int height_col,
    const int width_col,
    __global float* data_col,
    const int col_offset) {

    //Get the beginning of the data
    data_vol = data_vol + vol_offset;
    data_col = data_col + col_offset;

    int index = get_global_id(0);
    if (index < n) {
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int d_index = h_index / height_col;
        int d_out = d_index % depth_col;
        int channel_in = d_index / depth_col;

        int channel_out = channel_in * kernel_d * kernel_h * kernel_w;

        int d_in = d_out * stride_d - pad_d;
        int h_in = h_out * stride_h - pad_h;
        int w_in = w_out * stride_w - pad_w;

        __global float* data_col_ptr = data_col;

        #ifdef __VOL2IM_T
        data_col_ptr += channel_out + kernel_d*kernel_h*kernel_w*channels*
            ((d_out*height_col + h_out)*width_col + w_out);
        #else
        data_col_ptr += ((channel_out*depth_col +  d_out)*height_col + h_out)*width_col + w_out;
        #endif

        __global const float* data_vol_ptr = data_vol;
        data_vol_ptr += ((channel_in*depth + d_in)*height + h_in) * width + w_in;

        //Copy column of channels*volume
        for (int n = 0; n < kernel_d; ++n) {
            for (int i = 0; i < kernel_h; ++i) {
                for (int j = 0; j < kernel_w; ++j) {
                    int d = d_in + n;
                    int h = h_in + i;
                    int w = w_in + j;

                    *data_col_ptr = (d >= 0 && h >= 0 && w >= 0 &&
                                     d < depth && h < height && w < width) ?
                    //d*100 + h*10 + w : 0;
                    //channel_in*1000 + d_in*100 + h_in*10 + w_in : 0;
                    data_vol_ptr[(n*height + i)*width + j] : 0;

                    #ifdef __VOL2IM_T
                    data_col_ptr ++;
                    #else
                    data_col_ptr += depth_col * height_col * width_col;
                    #endif
                }
            }
        }
    }
}

#define BLOCK_SIZE 8
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void transpose(
    __global const float* data_src,
    const int src_offset,
    __global float* data_dst,
    const int dst_offset) {

    int gid_i = get_global_id(0);
    int gid_j = get_global_id(1);

    int height = get_global_size(0);
    int width = get_global_size(1);

    data_src = data_src + src_offset;
    data_dst = data_dst + dst_offset;

    if(gid_i < height && gid_j < width) {
        //data_dst[width * gid_i + gid_j] = data_src[height * gid_j + gid_i];
        data_dst[height * gid_j + gid_i] = data_src[width * gid_i + gid_j];
    }
}


__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void m2m(
    __global const float* data_a,
    __global float * data_b,
    const int b_offset,
    const int q,
    __global float* data_res,
    const int res_offset
    ) {
    //Identification of this workgroup
    int g_i = get_group_id(0);
    int g_j = get_group_id(1);

    //Identification of work-item
    int id_i = get_local_id(0);
    int id_j = get_local_id(1);

    //Matrixes dimensions
    int p = get_global_size(0);
    int r = get_global_size(1);

    int gid_i = get_global_id(0);
    int gid_j = get_global_id(1);

    //Get the beginning of the data
    data_b = data_b + b_offset;
    data_res = data_res + res_offset;

    if ((gid_i < (p/BLOCK_SIZE)*BLOCK_SIZE) &&
        (gid_j < (r/BLOCK_SIZE)*BLOCK_SIZE)) {
        //Process complete tiles

        //Number of submatrixes to be processed by each worker
        int ns = q / BLOCK_SIZE + (q % BLOCK_SIZE ? 1 : 0);
        float4 resp = (float4)(0, 0, 0, 0);
        __local float sub_a[BLOCK_SIZE][BLOCK_SIZE];
        __local float sub_b[BLOCK_SIZE][BLOCK_SIZE];

        for (int k=0; k < ns; k++) {
            //Copy submatrixes to local memory. Each worker copies one element
            unsigned int a_idx = q*(BLOCK_SIZE*g_i + id_i) + BLOCK_SIZE*k+id_j;
            sub_a[id_i][id_j] = a_idx < p*q ? data_a[a_idx] : 0;

            unsigned int b_idx = r*(BLOCK_SIZE*k + id_i) + BLOCK_SIZE*g_j+id_j;
            sub_b[id_i][id_j] = b_idx < q*r ? data_b[b_idx] : 0;

            barrier(CLK_LOCAL_MEM_FENCE);

            //#pragma unroll
            for (int k2 = 0; k2 < BLOCK_SIZE; k2+=4) {
                float4 line1=(float4)(sub_a[id_i][k2], sub_a[id_i][k2+1], sub_a[id_i][k2+2], sub_a[id_i][k2+3]);
                float4 line2=(float4)(sub_b[k2][id_j], sub_b[k2+1][id_j], sub_b[k2+2][id_j], sub_b[k2+3][id_j]);
                resp += line1 * line2;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        //unsigned int r_idx = BLOCK_SIZE*g_i + id_i + p*(BLOCK_SIZE*g_j+id_j);
        unsigned int r_idx = r*(BLOCK_SIZE*g_i + id_i) + BLOCK_SIZE*g_j + id_j;
        data_res[r_idx] = resp.x+resp.y+resp.z+resp.w;
    } else {
        //Process incomplete tiles
        //Так как ветвление происходит только в рамках одного wavefront (или группы?)
        //то по идее это повлияет на небольшое число workitemов
        if (gid_i < p && gid_j < r) { //Почему то иногда значения get_global_id больше get_global_size
            float resp = 0;           //Глюк или так и может быть?
            for (int k = 0; k < q; k++) {
                resp += data_a[q*gid_i + k] * data_b[r*k + gid_j];
            }
            data_res[r*gid_i + gid_j] = resp;
        }
    }
}

//Отличается тем, что вторая матрица должна быть транспонирована
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void m2tm(
    __global const float* data_a,
    __global float * data_b,
    const int b_offset,
    const int q,
    __global float* data_res,
    const int res_offset
    ) {
    //Identification of this workgroup
    int g_i = get_group_id(0);
    int g_j = get_group_id(1);

    //Identification of work-item
    int id_i = get_local_id(0);
    int id_j = get_local_id(1);

    //Matrixes dimensions
    int p = get_global_size(0);
    int r = get_global_size(1);

    int gid_i = get_global_id(0);
    int gid_j = get_global_id(1);

    //Get the beginning of the data
    data_b = data_b + b_offset;
    data_res = data_res + res_offset;

    if ((gid_i < (p/BLOCK_SIZE)*BLOCK_SIZE) &&
        (gid_j < (r/BLOCK_SIZE)*BLOCK_SIZE)) {
        //Process complete tiles

        //Number of submatrixes to be processed by each worker
        int ns = q / BLOCK_SIZE + (q % BLOCK_SIZE ? 1 : 0);
        float4 resp = (float4)(0, 0, 0, 0);
        __local float sub_a[BLOCK_SIZE][BLOCK_SIZE];
        __local float sub_b[BLOCK_SIZE][BLOCK_SIZE];

        for (int k=0; k < ns; k++) {
            //Copy submatrixes to local memory. Each worker copies one element
            unsigned int a_idx = q*(BLOCK_SIZE*g_i + id_i) + BLOCK_SIZE*k+id_j;
            sub_a[id_i][id_j] = a_idx < p*q ? data_a[a_idx] : 0;

            unsigned int b_idx = BLOCK_SIZE*k + id_i + q*(BLOCK_SIZE*g_j + id_j);
            sub_b[id_i][id_j] = b_idx < q*r ? data_b[b_idx] : 0;

            barrier(CLK_LOCAL_MEM_FENCE);

            //#pragma unroll
            for (int k2 = 0; k2 < BLOCK_SIZE; k2+=4) {
                float4 line1=(float4)(sub_a[id_i][k2], sub_a[id_i][k2+1], sub_a[id_i][k2+2], sub_a[id_i][k2+3]);
                float4 line2=(float4)(sub_b[k2][id_j], sub_b[k2+1][id_j], sub_b[k2+2][id_j], sub_b[k2+3][id_j]);
                resp += line1 * line2;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        //unsigned int r_idx = BLOCK_SIZE*g_i + id_i + p*(BLOCK_SIZE*g_j+id_j);
        unsigned int r_idx = r*(BLOCK_SIZE*g_i + id_i) + BLOCK_SIZE*g_j + id_j;
        data_res[r_idx] = resp.x+resp.y+resp.z+resp.w;
    } else {
        //Process incomplete tiles
        //Так как ветвление происходит только в рамках одного wavefront (или группы?)
        //то по идее это повлияет на небольшое число workitemов
        if (gid_i < p && gid_j < r) { //Почему то иногда значения get_global_id больше get_global_size
            float resp = 0;           //Глюк или так и может быть?
            for (int k = 0; k < q; k++) {
                //resp += data_a[q*gid_i + k] * data_b[r*k + gid_j];
                resp += data_a[q*gid_i + k] * data_b[k + q*gid_j];
            }
            data_res[r*gid_i + gid_j] = resp;
        }
    }
}
