//Image Recognition for Jetson-Nano
#include <jetson-inference/imageNet.h>
#include <jetson-utils/loadImage.h>

int main( int argc, char** argv)
{
	if (argc > 2)
	{
		printf("Expected image filename as an arguement");
		return 0;
	}

	const char* imgFilename = argv[1];

	uchar3* imgPtr = NULL; //shared CPU/GPU Pointer to image
	int imgWidth = 0;
	int imgHeight = 0;

	if (!loadImage(imgFilename, &imgPtr, &imgWidth, &imgHeight))
	{
		printf("failed to load image '%s'\n", imgFilename);
		return 0;
	}

	//Load an image network --> GoogLeNet with TensorRT
	imageNet* net = imageNet::Create(imageNet::GOOGLENET);

	if (!net)
	{
		printf("failed to load image recognition network\n");
		return 0;
	}
	//confidence of classification (0-1)
	unsigned float confidence = 0.0;

	//classify image, return object class index (-1 on error)
	const int classIndex = net->Classify(imgPtr,imgWidth,imgHeight, &confidence);

	// make sure a valid classification result was returned
	if( classIndex >= 0 )
	{
		// retrieve the name/description of the object class index
		const char* classDescription = net->GetClassDesc(classIndex);

		// print out the classification results
		printf("image is recognized as '%s' (class #%i) with %f%% confidence\n",
			  classDescription, classIndex, confidence * 100.0f);
	}
	else
	{
		// if Classify() returned < 0, an error occurred
		printf("failed to classify image\n");
	}

	// free the network's resources before shutting down
	delete net;
	return 0;

}
