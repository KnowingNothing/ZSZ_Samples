__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP;

__kernel void image_rotation(__read_only image2d_t input, __write_only image2d_t output, int M, int N, float theta) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    int cx = x / 2.0;
    int cy = y / 2.0;

    int px = x - cx;
    int py = y - cy;
    float pi = 3.1415926;

    float sin_theta = sin(pi / 180 * theta);
    float cos_theta = cos(pi / 180 * theta);

    float2 read_coord;
    read_coord.x = px * cos_theta - py * sin_theta + cx;
    read_coord.y = py * sin_theta + py * cos_theta + cy;

    float4 value;
    value = read_imagef(input, sampler, read_coord);
    write_imagef(output, (int2)(x, y), value);
}