#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>


#include <CL\cl.hpp>
#include "Utils.h"

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int averageTemperature(cl::Program program, cl::Buffer buffer_A, cl::Buffer buffer_B, cl::CommandQueue queue,
	size_t vector_size, size_t vector_elements, vector<int> outputFloat, size_t local_size)
{
	// Setup and execute the kernel (i.e. device code)
	cl::Kernel kernel_Average = cl::Kernel(program, "averageTemperature");//Average

	kernel_Average.setArg(0, buffer_A);
	kernel_Average.setArg(1, buffer_B);
	kernel_Average.setArg(2, cl::Local(1));

	queue.enqueueNDRangeKernel(kernel_Average, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size));

	//Copy the result from device to host
	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, vector_size, &outputFloat[0]);

	return outputFloat[0];
}

int minTemperature(cl::Program program, cl::Buffer buffer_A, cl::Buffer buffer_B, cl::CommandQueue queue,
	size_t vector_size, size_t vector_elements, vector<int> outputFloat, size_t local_size)
{
	// Setup and execute the kernel (i.e. device code)
	cl::Kernel kernel_Min = cl::Kernel(program, "minTemperature");//Minimum

	kernel_Min.setArg(0, buffer_A);
	kernel_Min.setArg(1, buffer_B);
	kernel_Min.setArg(2, cl::Local(1));

	queue.enqueueNDRangeKernel(kernel_Min, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size));

	//Copy the result from device to host
	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, vector_size, &outputFloat[0]);

	return outputFloat[0];
}

int maxTemperature(cl::Program program, cl::Buffer buffer_A, cl::Buffer buffer_B, cl::CommandQueue queue,
	size_t vector_size, size_t vector_elements, vector<int> outputFloat, size_t local_size)
{
	// Setup and execute the kernel (i.e. device code)
	cl::Kernel kernel_Max = cl::Kernel(program, "maxTemperature");//Maximum

	kernel_Max.setArg(0, buffer_A);
	kernel_Max.setArg(1, buffer_B);
	kernel_Max.setArg(2, cl::Local(1));

	queue.enqueueNDRangeKernel(kernel_Max, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size));

	//Copy the result from device to host
	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, vector_size, &outputFloat[0]);

	return outputFloat[0];
}

vector<int> hist_simple(cl::Program program, cl::Buffer buffer_A, cl::Buffer buffer_B, cl::CommandQueue queue,
				size_t vector_size, size_t vector_elements, vector<int> outputFloat, size_t local_size, int BidWidth)
{
	cl::Kernel kernel_Hist = cl::Kernel(program, "hist_simple");

	kernel_Hist.setArg(0, buffer_A);
	kernel_Hist.setArg(1, buffer_B);
	kernel_Hist.setArg(2, BidWidth);

	queue.enqueueNDRangeKernel(kernel_Hist, cl::NullRange, cl::NDRange(local_size), cl::NDRange(local_size));

	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, vector_size, &outputFloat[0]);

	return outputFloat;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "my_kernels.cl");

		cl::Program program(context, sources);

		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		typedef int mytype;
		std::vector<mytype> A(10, 1);

		//Part 4 - memory allocation

		vector<string> stationName;
		vector<int> year;
		vector<int> month;
		vector<int> day;
		vector<int> time;
		vector<int> temperature;
		string fileLocation;

		//All variables used to manage data entered from text file and store in relevant vectors
		string line;
		char searchItem = ' ';
		string word = "";
		int numOfChar = 0;
		int numOfSpace = 0;
		int temp = 0; //Used to store a temporary value of temp as atomic_add only works on integers.



			ifstream myfile("temp_lincolnshire_short.txt");
			if (myfile.is_open())
			{
				cout << "File Reading...\n";
				while (getline(myfile, line))
				{
					for (int i = 0; i < line.length(); i++)
					{
						word += line[i];
						if (line[i] == searchItem || i == line.length() - 1)
						{
							numOfSpace++;
							switch (numOfSpace)
							{
							case 1: 
								stationName.push_back(word);
								word = "";
								break;
							case 2: 
								year.push_back(stoi(word));
								word = "";
								break;
							case 3: 
								month.push_back(stoi(word));
								word = "";
								break;
							case 4: 
								day.push_back(stoi(word));
								word = "";
								break;
							case 5: 
								time.push_back(stoi(word));
								word = "";
								break;
							case 6: 
								temp = int(stof(word) * 10);
								temperature.push_back(temp);
								numOfSpace = 0;
								word = "";
								break;
							default: 
								break;
							}
						}
					}
					//cout << line << '\n';
				}
				cout << "Text file read in correctly\n";
				myfile.close();
		}
		
		size_t vector_elements = temperature.size();//number of elements
		size_t vector_size = temperature.size()*sizeof(int);//size in bytes

		size_t local_size = ( 64, 1 );
		size_t padding_size = temperature.size() % local_size;

		//host - output
		std::vector<int> outputList(vector_elements);

		std::vector<mytype> histOutput(vector_elements);
		size_t output_size = histOutput.size()*sizeof(mytype);//size in bytes

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, vector_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_output_size(context, CL_MEM_READ_WRITE, output_size);

		//Part 5 - device operations

		//5.1 Copy arrays A and B to device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &temperature[0]);
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, vector_size, &outputList[0]);
		queue.enqueueFillBuffer(buffer_output_size, 0, 0, output_size);//zero B buffer on device memory


		float output = 0.0f;
		float MaxV = 0.0f;
		float MinV = 0.0f;
		int runningTotal = 0;
		int BinWidth = 50;

				output = (float)(averageTemperature(program, buffer_A, buffer_B, queue, vector_size, vector_elements, outputList, local_size));
				output /= 10.0f;
				output = output / vector_elements;
				std::cout << "The average temperature is = " << output << std::endl;


				output = (float)(minTemperature(program, buffer_A, buffer_B, queue, vector_size, vector_elements, outputList, local_size));
				output /= 10.0f;
				std::cout << "The minimum temperature is = " << output << std::endl;
				MinV = output;

				output = (float)(maxTemperature(program, buffer_A, buffer_B, queue, vector_size, vector_elements, outputList, local_size));
				output /= 10.0f;
				std::cout << "The maximum temperature is = " << output << std::endl;
				MaxV = output;


				//if the input vector is not a multiple of the local_size
				//insert additional neutral elements (0 for addition) so that the total will not be affected
				if (padding_size) {
					//create an extra vector with neutral values
					std::vector<int> A_ext(local_size - padding_size, 100000);
					//append that extra vector to our input
					temperature.insert(temperature.end(), A_ext.begin(), A_ext.end());
				}

				outputList = (hist_simple(program, buffer_A, buffer_output_size, queue, vector_size, vector_elements, outputList, local_size, BinWidth));
				std::cout << "Hist " << outputList << std::endl;
				
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	std::cin.get();

	std::cin.get();
	return 0;
}