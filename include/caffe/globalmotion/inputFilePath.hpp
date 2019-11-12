#ifndef INPUTFILEPATH_HPP
#define INPUTFILEPATH_HPP

#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>
#include <map>

void checkInPutDir(std::string &s);

// InputFilePath(const std::string inputDir, const std::string fileType)
// input: inputDir that contains the searched files, fileType, defined the file type
// Check the inputDir also the subDir and add the files' names and files' pathes to the map 'filePath' and the vector 'fileName'
class InputFilePath
{
public:
	InputFilePath(){}
	InputFilePath(const std::string s, const std::string t) : inputDir(s), fileType(t) {}
	~InputFilePath(){}

	// Initial the class 'InputFilePath'
	// return value:
	// 0 - normal, 1 - failed
	int initial();
	
	//std::map<std::string, std::string> filePath;
	std::vector<std::string> fileName;
	std::string inputDir;
	std::string fileType;
	int numFiles;
	//bool CHECKSUBDIR;
private:
	// Check the input directory and save the information of the file in defined file type into 'filePath' and 'fileName'
	int getFileInfo(const std::string &s);
	// Check the directory path, change '\' to '/' and add '/' to the end if it not exist
};

#endif
