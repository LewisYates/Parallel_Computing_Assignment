//KERNEL GPU OPERATIONS

//kernel operation for average temperature calculation 
__kernel void avgTemp(__global const int* temperature, __global int* output, __local int* scratch){ 
	int id = get_global_id(0); //uniquely identifies a work-item (id)
	int lid = get_local_id(0);  //specifies a unique work-item ID within a given work-group (lid)
	int N = get_local_size(0);

	//cache all values from global memory to local memory
	scratch[lid] = temperature[id];

	barrier(CLK_LOCAL_MEM_FENCE); //wait for all local threads to finish copying from global to local

	for (int i = 1; i < N; i *= 2) { 
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE); //barrier is used to prevent other work-items from executing before all items have been processed
	}

	if (!lid) {
		atom_add(&output[0],scratch[lid]); //atomic operations lock resources so no interrupts can occur

	}
}

//kernel operation for maximum temperature calculation 
__kernel void maxTemp(__global const int* temperature, __global int* output, __local int* scratch){ 
	int id = get_global_id(0); 
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all values from global memory to local memory
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

//kernel operation for minimum temperature calculation 
__kernel void minTemp(__global const int* temperature, __global int* output, __local int* scratch){ 
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all values from global memory to local memory
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

//kernel operation for histogram implementation
__kernel void hist_auto(__global const int* temperature, __global int* output, int bincount, int minval, int maxval) { 
	int id = get_global_id(0);
	int temp_check = temperature[id]; //get temperature value from memory
	int range = maxval - minval; //range = maximum temp value - minimum temp value
	int i = temp_check; //set temperature value as counter
	int n = 0;
	int increment = range/bincount; //range divided by the user specified bin count
	int compareval = minval + increment; //From minimal value of dataset, increment and then.. 

	//add the specified bins when > compareval
	while (i > compareval)
	{
		compareval += increment;
		n++;
	}
	atomic_inc(&output[n]);
}


