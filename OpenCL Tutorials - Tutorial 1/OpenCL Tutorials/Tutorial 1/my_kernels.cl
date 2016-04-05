__kernel void averageTemperature(__global const int* temperature, __global int* output, __local int* scratch){ 
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

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) {
		atom_add(&output[0],scratch[lid]);

	}
}

__kernel void maxTemperature(__global const int* temperature, __global int* output, __local int* scratch){ 
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

__kernel void minTemperature(__global const int* temperature, __global int* output, __local int* scratch){ 
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

//a very simple histogram implementation
__kernel void hist_simple(__global const int* temperature, __global int* output, int bincount, int minval, int maxval) { 
	int id = get_global_id(0);
	int bin_index = temperature[id];
	int range = maxval-minval;
	int i = bin_index;
	int n = 0;
	int increment = range/bincount;
	int topBound = maxval - increment;
	int compareval = minval + increment;
	while (i > compareval)
	{
		compareval += increment;
		n++;
	}
	atomic_inc(&output[n]);
//	while (i <= (topBound))
//	{
//	i += increment;
//	n++;
//	}
//	n = bincount - n;
//	atomic_inc(&output[n]);

//	if (bin_index < 0) { atomic_inc(&output[0]);}
//	else if (bin_index >= 0 && bin_index < 100){  atomic_inc(&output[1]);}
//	else if (bin_index >= 100 && bin_index < 200){  atomic_inc(&output[2]);}
//	else if (bin_index >= 200 && bin_index < 300){  atomic_inc(&output[3]);}
//	else if (bin_index >= 300 && bin_index < 400){  atomic_inc(&output[4]);}
}


