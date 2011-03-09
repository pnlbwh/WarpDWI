#include <cstdlib>
#include <iostream>
#include <math.h>

using namespace std;

struct Matrix
{
    float e[16];              
    
    inline float& operator[](int index) {return e[index];}
    inline const float& operator[](int index) const {return e[index];}
};

struct Vector
{
    float e[4];              
    
    inline float& operator[](int index) {return e[index];}
    inline const float& operator[](int index) const {return e[index];}
    
    void normalize(void)
    {
        float length = sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
        e[0]/=length;
        e[1]/=length;
        e[2]/=length;
    }
    
};

ostream& operator<<(ostream& out, Matrix &m);
ostream& operator<<(ostream& out, Vector &v);

Matrix buildidentitymatrix(void);
Matrix buildrotationmatrix(float theta, float x, float y, float z);
Matrix buildscalematrix(float x, float y, float z);
Matrix buildtranslationmatrix(float x, float y, float z);

Vector buildvector(float x, float y, float z);

Matrix operator*(const Matrix &a, const Matrix &b);
void operator*=(Matrix &a, const Matrix &b);
Vector operator*(const Matrix &a, const Vector &v);
Vector operator/(const Vector &v, const float &f);

ostream& operator<<(ostream& out, const Matrix &m)
{
    out << m[0] << " " << m[4] << " " <<  m[8] << " " <<  m[12] << endl;
    out << m[1] << " " <<  m[5] << " " <<  m[9] << " " <<  m[13] << endl;
    out << m[2] << " " <<  m[6] << " " <<  m[10] << " " <<  m[14] << endl;
    out << m[3] << " " <<  m[7] << " " <<  m[11] << " " <<  m[15];
    return out;         
}

ostream& operator<<(ostream& out, Vector &v)
{
    out << "(" << v[0] << ", " << v[1] << ", " <<  v[2] << ")";
    return out;         
}

Matrix buildidentitymatrix(void)
{
    Matrix m;       
    m[0] = 1;
    m[1] = 0;
    m[2] = 0;
    m[3] = 0;
    
    m[4] = 0;
    m[5] = 1;
    m[6] = 0;
    m[7] = 0;
    
    m[8] = 0;
    m[9] = 0;
    m[10] = 1;
    m[11] = 0;
    
    m[12] = 0;
    m[13] = 0;
    m[14] = 0;
    m[15] = 1;     
    
    return m;
}

Matrix buildrotationmatrix(float theta, float x, float y, float z)
{
    Matrix m;       
    float c = cos(theta);
    float s = sin(theta);
    float t = 1 - cos(theta);     
    m[0] = t*x*x + c;
    m[1] = t*x*y + s*z;
    m[2] = t*x*z - s*y;
    m[3] = 0;
    
    m[4] = t*x*y - s*z;
    m[5] = t*y*y + c;
    m[6] = t*y*z + s*x;
    m[7] = 0;
    
    m[8] = t*x*z + s*y;
    m[9] = t*y*z - s*x;
    m[10] = t*z*z + c;
    m[11] = 0;
    
    m[12] = 0;
    m[13] = 0;
    m[14] = 0;
    m[15] = 1;          
    
    return m;
}

Matrix buildscalematrix(float x, float y, float z)
{
    Matrix m;     
    
    m[0] = x;
    m[1] = 0;
    m[2] = 0;
    m[3] = 0;
    
    m[4] = 0;
    m[5] = y;
    m[6] = 0;
    m[7] = 0;
    
    m[8] = 0;
    m[9] = 0;
    m[10] = z;
    m[11] = 0;
    
    m[12] = 0;
    m[13] = 0;
    m[14] = 0;
    m[15] = 1;     
    
    return m;
}

Matrix buildtranslationmatrix(float x, float y, float z)
{
    Matrix m;
    m[0] = 1;
    m[1] = 0;
    m[2] = 0;
    m[3] = 0;
    
    m[4] = 0;
    m[5] = 1;
    m[6] = 0;
    m[7] = 0;
    
    m[8] = 0;
    m[9] = 0;
    m[10] = 1;
    m[11] = 0;
    
    m[12] = x;
    m[13] = y;
    m[14] = z;
    m[15] = 1;     
    return m;
}

Matrix operator*(const Matrix &a, const Matrix &b)
{    
    Matrix m;     
    
    m[0] = b[0]*a[0] + b[1]*a[4] + b[2]*a[8] + b[3]*a[12];
    m[1] = b[0]*a[1] + b[1]*a[5] + b[2]*a[9] + b[3]*a[13];
    m[2] = b[0]*a[2] + b[1]*a[6] + b[2]*a[10] + b[3]*a[14];
    m[3] = b[0]*a[3] + b[1]*a[7] + b[2]*a[11] + b[3]*a[15];
    
    m[4] = b[4]*a[0] + b[5]*a[4] + b[6]*a[8] + b[7]*a[12];
    m[5] = b[4]*a[1] + b[5]*a[5] + b[6]*a[9] + b[7]*a[13];
    m[6] = b[4]*a[2] + b[5]*a[6] + b[6]*a[10] + b[7]*a[14];
    m[7] = b[4]*a[3] + b[5]*a[7] + b[6]*a[11] + b[7]*a[15];
    
    m[8] = b[8]*a[0] + b[9]*a[4] + b[10]*a[8] + b[11]*a[12];
    m[9] = b[8]*a[1] + b[9]*a[5] + b[10]*a[9] + b[11]*a[13];
    m[10] = b[8]*a[2] + b[9]*a[6] + b[10]*a[10] + b[11]*a[14];
    m[11] = b[8]*a[3] + b[9]*a[7] + b[10]*a[11] + b[11]*a[15];
    
    m[12] = b[12]*a[0] + b[13]*a[4] + b[14]*a[8] + b[15]*a[12];
    m[13] = b[12]*a[1] + b[13]*a[5] + b[14]*a[9] + b[15]*a[13];
    m[14] = b[12]*a[2] + b[13]*a[6] + b[14]*a[10] + b[15]*a[14];
    m[15] = b[12]*a[3] + b[13]*a[7] + b[14]*a[11] + b[15]*a[15];     
    
    return m;
}


void operator*=(Matrix &a, const Matrix &b)
{
    a = a*b;
}

Vector buildvector(float x, float y, float z)
{
    Vector v;
    v[0] = x;
    v[1] = y;
    v[2] = z;
    v[3] = 1;
    return v;              
}

Vector operator*(const Matrix &a, const Vector &v)
{
    Vector vnew;
    
    vnew[0] = v[0]*a[0] + v[1]*a[4] + v[2]*a[8] + v[3]*a[12];
    vnew[1] = v[0]*a[1] + v[1]*a[5] + v[2]*a[9] + v[3]*a[13];
    vnew[2] = v[0]*a[2] + v[1]*a[6] + v[2]*a[10] + v[3]*a[14];
    vnew[3] = v[0]*a[3] + v[1]*a[7] + v[2]*a[11] + v[3]*a[15];
    
    return vnew;
}

Vector operator/(const Vector &v, const float &f)
{
    return buildvector(v[0]/f, v[1]/f, v[2]/f);       
}

bool operator==(const Vector &v1, const Vector &v2)
{
  return (v1[0] == v2[0] && v1[1] == v2[1] && v1[2] == v2[2]);
}
bool operator<(const Vector &v1, const Vector &v2)
{
  return (v1[0] < v2[0]);
}
bool operator>(const Vector &v1, const Vector &v2)
{
  return (v1[0] > v2[0]);
}
