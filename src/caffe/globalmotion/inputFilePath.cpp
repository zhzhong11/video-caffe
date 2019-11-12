#include "caffe/globalmotion/inputFilePath.hpp"
#include <dirent.h>

int InputFilePath::initial()
{
	std::cout<<"======== Check "<<fileType<<" in \""<<inputDir<<"\" ========"<<std::endl;
	//printf("%5s Check %5s  in \"%20s\" %5s\n ", std::string(5, '=').c_str(), fileType.c_str(), inputDir.c_str(), std::string(5, '=').c_str());
	/*
	if (flag)
	{
		printf("%4s CHECK SUB DIRECTORY FUNCTION IS OPEN!! %4s\n", std::string(4, '*').c_str(), std::string(4, '*').c_str());
	}else{
		printf("%4s CHECK SUB DIRECTORY FUNCTION IS CLOSED!! %4s\n", std::string(4, '*').c_str(), std::string(4, '*').c_str());
	}*/

	checkInPutDir(inputDir);
	if (!getFileInfo(inputDir))
	{
		std::cout << "Check Files in: \"" << inputDir << "\" Succeed. And fileType is: \"" << fileType << "\"." << std::endl;
	}else{
		std::cout << "Check Files in: \"" << inputDir << "\" Failed. And fileType is: \"" << fileType << "\"." << std::endl;
		return 1;
	}
	
	if (fileName.empty())
	{
		std::cerr << "No Files Found in \"" << inputDir << "\"." << std::endl;
		numFiles=0;
		return 1;
	}else{
		std::cout<<"======== Search File Complete, Total "<<fileName.size()<<" Found. ========"<<std::endl;
		//printf("%4s Search File Complete, Total %5d Found. %4s\n", std::string(4, '=').c_str(), fileName.size(), std::string(4, '=').c_str());
		numFiles=fileName.size();
		return 0;
	}
}

int InputFilePath::getFileInfo(const std::string &currentDir)
{
	if (currentDir.empty())
	{
		std::cerr << "Initial inputDir Failed, Input Path is empty." << std::endl;
		return 1;
	}else{
		std::cout << "Initial inputDir, Input Path: \"" << currentDir << "\"." << std::endl;
	}


	struct dirent* ptr;
	DIR *dir;
	dir=opendir(currentDir.c_str());

	std::string fn;

	while((ptr=readdir(dir))!=NULL)
	{
		if(ptr->d_name[0]=='.')
			continue;
		fn.assign(ptr->d_name);

		std::string::size_type sz = fn.size();
		std::string::size_type pos = fn.rfind('.', std::string::npos);

		if (pos != std::string::npos)
		{
			if (fn.compare(pos + 1, sz - pos - 1, fileType) == 0)
			{			
				fileName.push_back(fn.substr(0, pos));
			}
		}
	}
	closedir(dir);
	
	return 0;
}

void checkInPutDir(std::string &filePath)
{
	if (!filePath.empty())
	{
		for (int i=0;i<filePath.size();++i){
			if (filePath[i] == '\\')
				filePath[i] = '/';
		}

		if (filePath[filePath.size() - 1] != '/')
			filePath.append("/");
	}	
}
