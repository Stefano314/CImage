#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <cmath>

using namespace std;

void SaveMatrix(string name, vector<vector<unsigned int>> matrix){
	ofstream file;
	file.open(name);
	for(unsigned int i=0; i<matrix.size(); i++){
		for(unsigned int j=0; j<matrix[0].size(); j++){
			file << matrix[i][j] << " ";
		}
		if(i<matrix.size()-1) file<<endl;
	}
	file.close();
}
	
class cImage {
private:
	// Private Methods
	unsigned int Get_ypixels(string filename);
	unsigned int Get_xpixels(string filename);
	
public:
	unsigned int x_pixels, y_pixels; //Attributes
	string image_name;
	vector<vector<unsigned int>> image; // GL matrix
	// Constructors & distructor
	cImage(){};
	~cImage(){};
	cImage(string filename){
		x_pixels = Get_xpixels(filename);
		y_pixels = Get_ypixels(filename);
		image_name = filename;
		image.resize(y_pixels, vector<unsigned int>(x_pixels));
		ifstream file;
		file.open(filename);
		string line;
		stringstream row;
		for(unsigned int i=0; i<y_pixels; i++){
			for(unsigned int j=0; j<x_pixels; j++){
				file >> image[i][j];
			}
		}
		file.close();
	}
	// Public Methods:
	// General
	void SaveImage(string name);
	void GetResolution();
	//Operations
	vector<vector<unsigned int>> SummedAreaTable();
	unsigned int LocalIntensity(vector<vector<unsigned int>>& tabsum, unsigned int x, unsigned int y, unsigned int window);
	cImage Thresholding(unsigned int window, float k);
};

unsigned int cImage::Get_ypixels(string filename){
	ifstream file(filename);
	int len = file.tellg(); // Starting position
	unsigned int rows = 0;
	string line;
	for(rows = 0; getline(file, line); rows++);
	file.seekg(len, std::ios_base::beg); // Go back to the initial position
	file.close();
	return rows;
}

unsigned int cImage::Get_xpixels(string filename){
	ifstream file(filename);
	unsigned int n = 0, val =0;
	int len = file.tellg(); // Starting position
	string line;
	getline(file, line); // Get string line
	file.seekg(len, std::ios_base::beg); // Go back to the initial position
	stringstream row; // Get stream
	row << line;    
	file.close();
	while(row >> val) n++;
	return n;
}

void cImage::GetResolution(){
	cout<<"- Image Resolution: "<<x_pixels<<"x"<<y_pixels<<endl;
}

vector<vector<unsigned int>> cImage::SummedAreaTable(){
	unsigned long int value = 0;
	vector<vector<unsigned int>> sat(y_pixels, vector<unsigned int>(x_pixels));
	for(unsigned int i = 0; i < y_pixels; i++) {
		for(unsigned int j = 0; j < x_pixels; j++) {
			value = image[i][j];
			if(i==0 && j==0)
				sat[i][j] = value;
			if(i == 0 && j > 0)
				sat[i][j] = value + sat[i][j-1];
			if(j == 0 && i > 0)
				sat[i][j] = value + sat[i-1][j]; 
			if(i > 0 && j > 0)
				sat[i][j] = value + sat[i][j-1] + sat[i-1][j] - sat[i-1][j-1];
		}
	}
	return sat;
}

unsigned int cImage::LocalIntensity(vector<vector<unsigned int>>& tabsum, unsigned int x, unsigned int y, unsigned int window=7){
	unsigned int d = round(window/2 + 0.1);
	return tabsum[x+d-1][y+d-1] + tabsum[x-d][y-d] - tabsum[x-d][y+d-1] - tabsum[x+d-1][y-d];
}

cImage cImage::Thresholding(unsigned int window=7, float k=0.2){
	cImage copy(image_name);
	double m = 0;
	vector<vector<unsigned int>> tabsum = copy.SummedAreaTable();
	unsigned int d = round(window/2. + 0.1);
	
	for(unsigned int i=0; i < d; i++){// Top line
		for(unsigned int j=0; j < x_pixels; j++){
			copy.image[i][j] = 0;
		}
	}
	for(unsigned int i=y_pixels-d; i < y_pixels; i++){// Bot line
		for(unsigned int j=0; j < x_pixels; j++){
			copy.image[i][j] = 0;
		}
	}
	for(unsigned int i=d; i < y_pixels-d; i++){ // Left vertical segment
		for(unsigned int j=0; j < d; j++){
			copy.image[i][j] = 0;
		}
	}
	for(unsigned int i=d; i < y_pixels-d; i++){ // Right vertical segment
		for(unsigned int j=x_pixels-d; j < x_pixels; j++){
			copy.image[i][j] = 0;
		}
	}	
	for(unsigned int i=d; i < y_pixels-d; i++){// y values, removing borders
		for(unsigned int j=d; j < x_pixels-d; j++){  // x values, removing borders
			m = LocalIntensity(tabsum, i, j, window)/(1.0*window*window);
			if(copy.image[i][j] >= m * (1 + k*((copy.image[i][j]-m)/(1-copy.image[i][j] + m) - 1)))
				copy.image[i][j] = 255;
			else
				copy.image[i][j] = 0;
		}
	}
	return copy;
}

void cImage::SaveImage(string name){
	ofstream file;
	file.open(name);
	for(unsigned int i=0; i<y_pixels; i++){
		for(unsigned int j=0; j<x_pixels; j++){
			file << image[i][j] << " ";
		}
		if(i<y_pixels-1) file<<endl;
	}
	file.close();
}
	
int main()
{
	string filename("smile.txt"); // Image name
	cImage pic(filename); // Load image
	//pic.GetResolution(); // Check image width
	cImage T_Image = pic.Thresholding(11,0.2);
	T_Image.GetResolution();
	T_Image.SaveImage("smile_threshold.txt");
	return 0;
}
