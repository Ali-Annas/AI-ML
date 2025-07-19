#include "include/utils.h"
int main()
{
	int x;
	int y;
	int i;
	vector<int> v1(6);
	vector<int> v2(6);
	vector<int> v3(6);
	x = 2;
	assign_vector(v1,2 * x);
	assign_vector(v2,vector<int>({1,1,2,2,3,3}));
	cout << "V1<dot>V2" << ": " << vector_dotprod(v1,v2) << endl;
	y = index_vector(v2,4);
	i = 0;
	for(int _ = 1; _ <= y; _++)
	{
		v1[i] = i;
		i = i + 1;
	}
	cout << "v1 is" << ": " << v1 << endl;
	cout << "v2 indexed" << ": " << index_vector(v2,v1) << endl;
	cout << "that reversed" << ": " << index_vector(index_vector(v2,v1),vector<int>({5,4,3,2,1,0})) << endl;
	assign_vector(v3,v1 + v2);
	cout << "" << ": " << index_vector(v2,(vector_dotprod(vector<int>({2,1,0,2,2,0}),v3) / 10)) << endl;
	vector<int> a(3);
	assign_vector(a,vector<int>({10,0,20}));
	i = 0;
	for(int _ = 1; _ <= 3; _++)
	{
		if(vector_dotprod(a,vector<int>({1,0,0})))
		{
			cout << "Rotate" << ": " << i << a << endl;
			assign_vector(a,index_vector(a,vector<int>({2,0,1})));
		}
		i = i + 1;
	}
	vector<int> z(4);
	assign_vector(z,10);
	assign_vector(z,(z + vector<int>({2,4,6,8})) / 2);
	assign_vector(z,z - 3 + vector<int>({2,3,4,5}));
	cout << "z is" << ": " << z << endl;
	cout << "z summed" << ": " << vector_dotprod(z,vector<int>({1,1,1,1})) << endl;
}
