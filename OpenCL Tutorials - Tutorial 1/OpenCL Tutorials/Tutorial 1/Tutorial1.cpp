#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <unordered_map>
#include <CL\cl.hpp>
#include "Utils.h"

//Call to the kernel for min temperature function - minTemp
int minTemperature(cl::Program program, cl::Buffer buffer_A, cl::Buffer buffer_B, cl::CommandQueue queue,
	size_t vector_size, size_t vector_elements, vector<int> outputFloat, size_t local_size)
{
	// Setup and execute the kernel (i.e. device code)
	cl::Kernel kernel_Min = cl::Kernel(program, "minTemp");

	kernel_Min.setArg(0, buffer_A);
	kernel_Min.setArg(1, buffer_B);
	kernel_Min.setArg(2, cl::Local(1));
	queue.enqueueNDRangeKernel(kernel_Min, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size));

	//Copy the result from device to host
	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, vector_size, &outputFloat[0]);

	return outputFloat[0];
}

//Call to the kernel for max temperature function - maxTemp
int maxTemperature(cl::Program program, cl::Buffer buffer_A, cl::Buffer buffer_B, cl::CommandQueue queue,
	size_t vector_size, size_t vector_elements, vector<int> outputFloat, size_t local_size)
{
	cl::Kernel kernel_Max = cl::Kernel(program, "maxTemp");
	kernel_Max.setArg(0, buffer_A);
	kernel_Max.setArg(1, buffer_B);
	kernel_Max.setArg(2, cl::Local(1));
	queue.enqueueNDRangeKernel(kernel_Max, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size));
	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, vector_size, &outputFloat[0]);

	return outputFloat[0];
}

//Call to the kernel for average temperature function - avgTemp
int averageTemperature(cl::Program program, cl::Buffer buffer_A, cl::Buffer buffer_B, cl::CommandQueue queue,
	size_t vector_size, size_t vector_elements, vector<int> outputFloat, size_t local_size)
{
	cl::Kernel kernel_Average = cl::Kernel(program, "avgTemp");
	kernel_Average.setArg(0, buffer_A);
	kernel_Average.setArg(1, buffer_B);
	kernel_Average.setArg(2, cl::Local(1));
	queue.enqueueNDRangeKernel(kernel_Average, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size));
	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, vector_size, &outputFloat[0]);

	return outputFloat[0];
}

//Call to the kernel for histogram function - hist_auto
vector<int> histogram(cl::Program program, cl::Buffer buffer_A, cl::Buffer buffer_B, cl::CommandQueue queue,
	size_t vector_size, size_t vector_elements, vector<int> outputFloat, size_t local_size, int count, int minval, int maxval)
{
	cl::Kernel kernel_Hist = cl::Kernel(program, "hist_auto");
	kernel_Hist.setArg(0, buffer_A);
	kernel_Hist.setArg(1, buffer_B);
	kernel_Hist.setArg(2, count);
	kernel_Hist.setArg(3, minval);
	kernel_Hist.setArg(4, maxval);
	queue.enqueueNDRangeKernel(kernel_Hist, cl::NullRange, cl::NDRange(vector_elements), cl::NDRange(local_size));
	queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, vector_size, &outputFloat[0]);

	return outputFloat;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

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
		vector<string> station;
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
		int space = 0;
		int temp = 0; //Used to store a temporary value of temp as atomic_add only works on integers.

			//read in the valid dataset 
			ifstream dataset("temp_lincolnshire_short.txt");
		
			//if dataset is open, read file line by line - specifying each seperate node {Station, Year, Month, Day, Time, Temperature} and putting into declared vector 
			if (dataset.is_open())
				cout << "Reading File...\n";
			{
				while (getline(dataset, line))
				{
					for (int i = 0; i < line.length(); i++)
					{
						word += line[i];
						if (line[i] == searchItem || i == line.length() - 1)
						{
							space++;
							switch (space)
							{
							case 1:
								station.push_back(word); //push_back is used to add each variable to the end of the specified vector
								word = "";
								break;
							case 2:
								year.push_back(stoi(word)); //stoi is used to convert a string to an int value
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
								temp = int(stof(word) * 10); //stof is used to convert a string to a float value
								temperature.push_back(temp);
								space = 0;
								word = "";
								break;
							default:
								break;
						}

					}
				}
			}
		}

		size_t vector_elements = temperature.size();//number of elements
		size_t vector_size = temperature.size()*sizeof(int);//size in bytes

		size_t local_size = (64, 1); 
		size_t padding_size = temperature.size() % local_size;
		
		//host - output
		std::vector<int> outputList(vector_elements);
		std::vector<mytype> histOutput(vector_elements);
		size_t output_size = histOutput.size()*sizeof(mytype);//size in bytes

		//device - buffers
		cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY, vector_size);
		cl::Buffer outputBuffer(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_output_size(context, CL_MEM_READ_WRITE, output_size);

		//Part 5 - device operations

		//5.1 Copy arrays input(A) and output(B) to device memory
		queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, vector_size, &temperature[0]); //write to input buffer
		queue.enqueueWriteBuffer(outputBuffer, CL_TRUE, 0, vector_size, &outputList[0]); //write to output buffer
		queue.enqueueFillBuffer(buffer_output_size, 0, 0, output_size);//zero output buffer on device memory

		float output = 0.0f;
		float maxVal = 0;
		float minVal = 0;

		//Calculate Average Temperature using GPU Kernel
		output = (float)(averageTemperature(program, inputBuffer, outputBuffer, queue, vector_size, vector_elements, outputList, local_size));
		output /= 10.0f;
		output = output / vector_elements;
		std::cout << std::setprecision(3) << "\nAverage Temperature: " << output << std::endl;

		//Calculate Minimum Temperature using GPU Kernel
		output = (float)(minTemperature(program, inputBuffer, outputBuffer, queue, vector_size, vector_elements, outputList, local_size));
		minVal = output;
		output /= 10.0f;
		std::cout << "Minimum Temperature: " << output << std::endl;

		//Calculate Maximum Temperature using GPU kernel
		output = (float)(maxTemperature(program, inputBuffer, outputBuffer, queue, vector_size, vector_elements, outputList, local_size));
		maxVal = output;
		output /= 10.0f;
		std::cout << "Maximum Temperature: " << output << std::endl;

			std::cout << "How Many Histogram Bins Do You Want To Display? " << endl;
			int binNo = 1;
			cin >> binNo;

			//while user entered bin number is 0 or a string (value other than int) ERROR. 
			while (binNo <= 0 || binNo > 200 || cin.fail()) 
			{
				cin.clear();
				cin.ignore(10000, '\n');
				std::cout << "Invalid Entry, Please Enter a Numerical Value >= 1" << endl;
				cin >> binNo;
			}

				//Calculate Histogram Bins using GPU kernel
				outputList = (histogram(program, inputBuffer, buffer_output_size, queue, vector_size, vector_elements, outputList, local_size, binNo, minVal, maxVal));
				std::cout << "\n" << std::endl;
				float binIncrement = ((maxVal - minVal) / binNo);
				std::cout << "  Min \t         Max         No. Values" << std::endl;
				std::cout << " _____          _____        __________ \n" << std::endl;
				 
				//Calculates the Minimum and Maximum Range of Histogram with a user defined bin value and the total in that bin
				for (int i = 1; i < binNo + 1; i++)
				{
					std::cout << std::fixed << std::setprecision(2) << "  " << ((minVal + ((i - 1)*binIncrement)) / 10) << "    \t" << ((minVal + (i*binIncrement)) / 10) << " \t   =   \t" << (outputList[i - 1]) << std::endl;
				}
			}

	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	string userExit;
	std::cout << "\n Press Enter To Exit..." << std::endl;
	
	std::cin.get();
	std::cin.get();

	return 0;
}