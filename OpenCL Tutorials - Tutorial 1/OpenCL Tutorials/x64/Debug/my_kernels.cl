__kernel void avgTemp(__global const int* temperature, __global int* output, __local int* scratch){ 
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cace all values from global memory to local memory
	scratch[lid] = temperature[id];

	barrier(CLK_LOCAL_MEM_FENCE); //wait for all local threads to finish copying from global to local

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid) {
		atom_add(&output[0],scratch[lid]);

	}
}

__kernel void maxTemp(__global const int* temperature, __global int* output, __local int* scratch){ 
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cace all values from global memory to local memory
	scratch[lid] = temperature[id];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) {
		if (!(scratch[lid] >= scratch[lid + i]))
			scratch[lid] = scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

		if (!lid){
				atom_max(&output[0],scratch[lid]);
		}
}

__kernel void minTemp(__global const int* temperature, __global int* output, __local int* scratch){ 
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cace all values from global memory to local memory
	scratch[lid] = temperature[id];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2){
		if (!(scratch[lid] <= scratch[lid + i]))
			scratch[lid] = scratch[lid + i];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid){
				atom_min(&output[0],scratch[lid]);
		}
}

//histogram implementation
__kernel void hist_auto(__global const int* temperature, __global int* output, int bincount, int minval, int maxval) { 
	int id = get_global_id(0);
	int bin_index = temperature[id];
	int range = maxval-minval;
	int i = bin_index;
	int n = 0;
	int increment = range/bincount;
	int compareval = minval + increment;
	while (i > compareval)
	{
		compareval += increment;
		n++;
	}
	atomic_inc(&output[n]);
}


