#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace std;

vector<int> operator+(const vector<int>& a,const vector<int>& b)
{
    if(a.size() != b.size())
        throw std::logic_error("vector sizes not equal");
    vector<int> result;
    for(int i = 0; i < a.size(); i++)
        result.push_back(a[i] + b[i]);
    return result;
}
vector<int> operator-(const vector<int>& a,const vector<int>& b)
{
    if(a.size() != b.size())
        throw std::logic_error("vector sizes not equal");
    vector<int> result;
    for(int i = 0; i < a.size(); i++)
        result.push_back(a[i] - b[i]);
    return result;
}
vector<int> operator*(const vector<int>& a,const vector<int>& b)
{
    if(a.size() != b.size())
        throw std::logic_error("vector sizes not equal");
    vector<int> result;
    for(int i = 0; i < a.size(); i++)
        result.push_back(a[i] * b[i]);
    return result;
}
vector<int> operator/(const vector<int>& a,const vector<int>& b)
{
    if(a.size() != b.size())
        throw std::logic_error("vector sizes not equal");
    vector<int> result;
    for(int i = 0; i < a.size(); i++)
        result.push_back(a[i] / b[i]);
    return result;
}
int vector_dotprod(const vector<int>& a,const vector<int>& b)
{
    if(a.size() != b.size())
        throw std::logic_error("vector sizes not equal");
    int result = 0;
    for(int i = 0; i < a.size(); i++)
        result += (a[i] * b[i]);
    return result;
}
vector<int> operator+(const vector<int>& a,int val)
{
    vector<int> result;
    for(int i = 0; i < a.size(); i++)
        result.push_back(a[i] + val);
    return result;
}
vector<int> operator-(const vector<int>& a,int val)
{
    vector<int> result;
    for(int i = 0; i < a.size(); i++)
        result.push_back(a[i] - val);
    return result;
}
vector<int> operator/(const vector<int>& a,int val)
{
    vector<int> result;
    for(int i = 0; i < a.size(); i++)
        result.push_back(a[i] / val);
    return result;
}
vector<int> operator*(const vector<int>& a,int val)
{
    vector<int> result;
    for(int i = 0; i < a.size(); i++)
        result.push_back(a[i] * val);
    return result;
}

void assign_vector(vector<int>& v,int val)
{
    v.assign(v.size(),val);
}
void assign_vector(vector<int>& v,const vector<int>& val)
{
    v = val;
}

vector<int> index_vector(const vector<int>& a,const vector<int>& b)
{
    if(a.size()!=b.size())
    {
        throw logic_error("vector sizes not same");
    }
    vector<int> result;
    for(int i = 0; i < b.size(); i++)
    {
        int idx = b[i];
        if(idx < a.size())
            result.push_back(a[idx]);
    }
    return result;
}
int index_vector(const vector<int>& a,size_t idx)
{
    if(idx < a.size())
        return a[idx];
    return 0;
}
ostream& operator<<(ostream& out,const vector<int>& a)
{
    out << '[';
    int i = 0;
    for(auto e: a)
    {
        out << e;
        if(i != a.size()-1)
            out << ',';
        i++;
    }
    out << ']';
    return out;
}
