#include "WarpVolumeCLP.h"
#include <itkImage.h>
#include <itkOrientedImage.h>
#include <itkImageIOBase.h>
#include <itkImageFileReader.h>
#include <itkImageRegionIterator.h>
#include <itkVectorImage.h>
#include <itkVariableLengthVector.h>
#include <itkMetaDataObject.h>
#include <itkVectorContainer.h>
#include <vector>
#include <algorithm>
#include "itkWarpImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkExtractRotationsFilter.h"

#include "itkMultivariateLegendrePolynomial.h"
#include "SphericalHarmonicPolynomial.h"
#include <math.h>

#include "itkBinaryFunctorImageFilter.h"
#include "itkUnaryFunctorImageFilter.h"
//#include "vnl_qr.h"
#include "vnl_determinant.h"

#include "itkPluginFilterWatcher.h"
#include "itkPluginUtilities.h"
#include "vtkSmartPointer.h"
#include "vtkTeemEstimateDiffusionTensor.h"
#include "vtkMatrix4x4.h"
#include "vtkNRRDReader.h"
#include "vtkNRRDWriter.h"
#include "vtkMRMLNRRDStorageNode.h"
#include "vtkMath.h"
#include "vtkImageData.h"
#include "vtkDoubleArray.h"

#include "mat.h"

#include "matrixlib.h"
#include "SHFilter.h"

namespace
{

struct Triangle
{
  Vector v0, v1, v2;
  Vector normal;

  Triangle(Vector v0, Vector v1, Vector v2)
  {
    this->v0 = v0;
    this->v1 = v1;
    this->v2 = v2;
  }

  void setnormal(Vector normal)
  {
    this->normal = normal;
  }

  void setnormal(float x, float y, float z)
  {
    setnormal(buildvector(x, y, z));
  }

  Triangle(void) {}

};

struct parameters
{
  std::string input_image;
  std::string output_image;
  std::string warp;
  bool resample;
  bool resample_self;
  bool without_baselines;
  //std::string resultsDirectory;
};

void subdivide(vector<Triangle>& triangles, vector<Vector>& vertices)
{
    unsigned origSize = triangles.size();
    for (unsigned i = 0 ; i < origSize ; ++i)
    {
        Triangle& t = triangles[i];
        Vector a = t.v0;
        Vector b = t.v1;
        Vector c = t.v2;
        Vector v1 = buildvector(a[0]+b[0], a[1]+b[1], a[2]+b[2]);
        Vector v2 = buildvector(a[0]+c[0], a[1]+c[1], a[2]+c[2]);
        Vector v3 = buildvector(b[0]+c[0], b[1]+c[1], b[2]+c[2]);
        v1.normalize();
        v2.normalize();
        v3.normalize();
        t.v0 = v1; // overwrite the original
        t.v1 = v3; 
        t.v2 = v2; 
        triangles.push_back(Triangle(a, v1, v2));
        triangles.push_back(Triangle(c, v2, v3));
        triangles.push_back(Triangle(b, v3, v1));

       if ((std::find(vertices.begin(), vertices.end(), v1) != vertices.end()) == false)
         vertices.push_back(v1);

       if ((std::find(vertices.begin(), vertices.end(), v2) != vertices.end()) == false)
         vertices.push_back(v2);

       if ((std::find(vertices.begin(), vertices.end(), v3) != vertices.end()) == false)
         vertices.push_back(v3);
    }
}

vnl_matrix<double> sample_sphere_as_icosahedron(int levels)
{
    vector<Triangle> triangles;
    vector<Vector> vertices;
    
    // build an icosahedron
    
    float t = (1 + sqrt(5.0))/2.0;
    float s = sqrt(1 + t*t);
    // create the 12 vertices
    Vector v0 = buildvector(t, 1, 0)/s;
    Vector v1 = buildvector(-t, 1, 0)/s;
    Vector v2 = buildvector(t, -1, 0)/s;
    Vector v3 = buildvector(-t, -1, 0)/s;
    Vector v4 = buildvector(1, 0, t)/s;
    Vector v5 = buildvector(1, 0, -t)/s;    
    Vector v6 = buildvector(-1, 0, t)/s;
    Vector v7 = buildvector(-1, 0, -t)/s;
    Vector v8 = buildvector(0, t, 1)/s;
    Vector v9 = buildvector(0, -t, 1)/s;
    Vector v10 = buildvector(0, t, -1)/s;
    Vector v11 = buildvector(0, -t, -1)/s;
    vertices.push_back(v0);
    vertices.push_back(v1);
    vertices.push_back(v2);
    vertices.push_back(v3);
    vertices.push_back(v4);
    vertices.push_back(v5);   
    vertices.push_back(v6);
    vertices.push_back(v7);
    vertices.push_back(v8);
    vertices.push_back(v9);
    vertices.push_back(v10);
    vertices.push_back(v11);

    // create the 20 triangles
    triangles.push_back(Triangle(v0, v8, v4));
    triangles.push_back(Triangle(v1, v10, v7));
    triangles.push_back(Triangle(v2, v9, v11));
    triangles.push_back(Triangle(v7, v3, v1));    
    triangles.push_back(Triangle(v0, v5, v10));
    triangles.push_back(Triangle(v3, v9, v6));
    triangles.push_back(Triangle(v3, v11, v9));
    triangles.push_back(Triangle(v8, v6, v4));    
    triangles.push_back(Triangle(v2, v4, v9));
    triangles.push_back(Triangle(v3, v7, v11));
    triangles.push_back(Triangle(v4, v2, v0));
    triangles.push_back(Triangle(v9, v4, v6));    
    triangles.push_back(Triangle(v2, v11, v5));
    triangles.push_back(Triangle(v0, v10, v8));
    triangles.push_back(Triangle(v5, v0, v2));
    triangles.push_back(Triangle(v10, v5, v7));    
    triangles.push_back(Triangle(v1, v6, v8));
    triangles.push_back(Triangle(v1, v8, v10));
    triangles.push_back(Triangle(v6, v1, v3));
    triangles.push_back(Triangle(v11, v7, v5));
    
    for (int ctr = 0; ctr < levels; ctr++) 
    {
      subdivide(triangles, vertices);
      //subdivide(triangles);
    }
    
    sort(vertices.begin(), vertices.end());
    //std::cout << "num vertices is " << vertices.size() << std::endl;
    //for (int i=0; i < vertices.size(); i++)
    //{
      //printf("%f  %f  %f \n", vertices[i][0], vertices[i][1], vertices[i][2]);
    //}

    vnl_matrix<double> vertices_matrix(vertices.size(),3);
    for(unsigned int i = 0; i < vertices.size(); i++)
    {
      vertices_matrix(i,0) =  vertices[i][0];
      vertices_matrix(i,1) =  vertices[i][1];
      vertices_matrix(i,2) =  vertices[i][2];
    }
    return vertices_matrix;
}

/* 
 * This function was taken from ResampleVolume2.cxx
 *
 * Separate the vector image into a vector of images
 */
template< class PixelType >
int SeparateImages( const typename itk::VectorImage< PixelType , 3 >
                    ::Pointer &imagePile ,
                    std::vector< typename itk::OrientedImage< PixelType , 3 >::Pointer > &vectorImage
                  )
{
   typedef itk::OrientedImage< PixelType , 3 > ImageType;
   typedef itk::VectorImage< PixelType , 3 > VectorImageType;
   typename itk::VectorImage< PixelType , 3 >::SizeType size;
   typename itk::VectorImage< PixelType , 3 >::DirectionType direction;
   typename itk::VectorImage< PixelType , 3 >::PointType origin;
   typename itk::VectorImage< PixelType , 3 >::SpacingType spacing;
   size = imagePile->GetLargestPossibleRegion().GetSize();
   direction=imagePile->GetDirection();
   origin=imagePile->GetOrigin();
   spacing=imagePile->GetSpacing();
   typename itk::ImageRegionIterator< VectorImageType > in( imagePile , imagePile->GetLargestPossibleRegion() );
   typedef typename itk::ImageRegionIterator< ImageType > IteratorImageType;
   std::vector< IteratorImageType > out;
   for( unsigned int i = 0; i < imagePile->GetVectorLength(); i++ )
   {
      typename ImageType::Pointer imageTemp = ImageType::New();
      imageTemp->SetRegions( size );
      imageTemp->SetOrigin( origin );
      imageTemp->SetDirection( direction );
      imageTemp->SetSpacing( spacing );
      imageTemp->Allocate();
      vectorImage.push_back( imageTemp );
      IteratorImageType outtemp( imageTemp , imageTemp->GetLargestPossibleRegion() );
      outtemp.GoToBegin();
      out.push_back( outtemp );
   }
   for( in.GoToBegin(); !in.IsAtEnd(); ++in )
   {
      itk::VariableLengthVector< PixelType > value = in.Get();
      for( unsigned int i = 0; i < imagePile->GetVectorLength(); i++ )
      {
         out[ i ].Set( value[ i ] );
         ++out[ i ];
      }
   }
   return EXIT_SUCCESS;
}

/*
 * This function was taken from ResampleVolume2.cxx
 *
 * Write back the vector of images into a image vector
 */
template<class PixelType>
int AddImage( typename itk::VectorImage< PixelType, 3 >
              ::Pointer &imagePile,
              const std::vector< typename itk::OrientedImage< PixelType , 3 > ::Pointer > &vectorImage
            )
{
   typedef itk::OrientedImage< PixelType , 3 > ImageType;
   imagePile->SetRegions( vectorImage.at( 0 )->GetLargestPossibleRegion().GetSize() );
   imagePile->SetOrigin( vectorImage.at( 0 )->GetOrigin() );
   imagePile->SetDirection( vectorImage.at( 0 )->GetDirection() );
   imagePile->SetSpacing( vectorImage.at( 0 )->GetSpacing() );
   imagePile->SetVectorLength( vectorImage.size() );
   imagePile->Allocate();
   typename itk::ImageRegionIterator< itk::VectorImage< PixelType , 3 > > out( imagePile ,
                                                                               imagePile->GetLargestPossibleRegion()
                                                                             );
   typedef typename itk::ImageRegionIterator< ImageType > IteratorImageType;
   std::vector< IteratorImageType > in;
   for( unsigned int i = 0; i < imagePile->GetVectorLength(); i++ )
   {
      IteratorImageType intemp( vectorImage.at( i ) , vectorImage.at( i )->GetLargestPossibleRegion() );
      intemp.GoToBegin();
      in.push_back( intemp );
   }
   itk::VariableLengthVector< PixelType > value;
   value.SetSize( vectorImage.size() );
   for( out.GoToBegin(); !out.IsAtEnd(); ++out )
   {
      for( unsigned int i = 0; i < imagePile->GetVectorLength(); i++ )
      {
         value.SetElement( i , in.at( i ).Get() );
         ++in[ i ];
      }
    out.Set( value );
    }
  return EXIT_SUCCESS;
}

void PrintMatrixRow(vnl_matrix<double> matrix, int row)
{
  std::cout << "matrix is " << matrix.rows() << " by " << matrix.columns() << std::endl;
  for (unsigned int i = 0; i < matrix.cols(); i ++)
  {
    std::cout << matrix(row, i) << std::endl;
  }
  std::cout  << std::endl;
}

void PrintMatrix(vnl_matrix<double> matrix, int col)
{
  std::cout << "matrix is " << matrix.rows() << " by " << matrix.columns() << std::endl;
  for (unsigned int i = 0; i < matrix.rows(); i ++)
  {
    std::cout << matrix(i, col) << std::endl;
  }
  std::cout  << std::endl;
}

void PrintVector(vnl_vector<double> vector)
{
  for (unsigned int i = 0; i < vector.size(); i ++)
  {
    std::cout << vector(i) << std::endl;
  }
  std::cout  << std::endl;
}

struct header_struct {
  vnl_matrix<double> gradients;
  unsigned int numberOfBaselineImages;
};

void PrintDictionary(itk::MetaDataDictionary& dico)
{
  std::vector<std::string> imgMetaKeys = dico.GetKeys();
  std::vector<std::string>::const_iterator itKey = imgMetaKeys.begin();
  std::string metaString;

  std::cout << "=====================================================================" << std::endl;
  std::cout << "The input image's dictionary header" << std::endl;
  std::cout << "=====================================================================" << std::endl;
  for (; itKey != imgMetaKeys.end(); itKey ++)
  {
    itk::ExposeMetaData<std::string> (dico, *itKey, metaString);
    std::cout << *itKey << " ---> " << metaString << std::endl;
  }
  std::cout << "=====================================================================" << std::endl;
}

header_struct GetGradients(itk::MetaDataDictionary& imgMetaDictionary)
{
  unsigned int numberOfGradientImages = 0;
  unsigned int numberOfImages = 0;
  bool readb0 = false;
  double b0 = 0;

  typedef vnl_vector_fixed<double,3> GradientDirectionType;
  GradientDirectionType vec3d;
  typedef itk::VectorContainer<unsigned int, GradientDirectionType> GradientDirectionContainerType;
  GradientDirectionContainerType::Pointer diffusionVectors = GradientDirectionContainerType::New(); 

  std::vector<std::string> imgMetaKeys = imgMetaDictionary.GetKeys();
  std::vector<std::string>::const_iterator itKey = imgMetaKeys.begin();
  std::string metaString;

  for (; itKey != imgMetaKeys.end(); itKey ++)
    {
    double x,y,z;

    itk::ExposeMetaData<std::string> (imgMetaDictionary, *itKey, metaString);
    if (itKey->find("DWMRI_gradient") != std::string::npos)
    {
      sscanf(metaString.c_str(), "%lf %lf %lf\n", &x, &y, &z);
      vec3d[0] = x; vec3d[1] = y; vec3d[2] = z;
      diffusionVectors->InsertElement( numberOfImages, vec3d );
      ++numberOfImages;
      if (vec3d[0] == 0.0 && vec3d[1] == 0.0 && vec3d[2] == 0.0)  //baseline image
      {
        continue;
      }
      ++numberOfGradientImages;
    }
    else if (itKey->find("DWMRI_b-value") != std::string::npos)
    {
      readb0 = true;
      b0 = atof(metaString.c_str());
    }
  }
  std::cout << "Number of gradient images: "
            << numberOfGradientImages
            << " and Number of baseline images: "
            << numberOfImages - numberOfGradientImages
            << std::endl;
  if(!readb0)
  {
    std::cerr << "BValue not specified in header file" << std::endl;
  }

  vnl_matrix<double> gradients(numberOfGradientImages, 3);
  unsigned int gradientIndex = 0;
  for (unsigned int i = 0; i < numberOfImages; i++)
  {
    if (diffusionVectors->ElementAt(i).two_norm() <= 0.0) // this is a baseline image
      continue;
    gradients(gradientIndex,0) = (diffusionVectors->ElementAt(i))[0];
    gradients(gradientIndex,1) = (diffusionVectors->ElementAt(i))[1];
    gradients(gradientIndex,2) = (diffusionVectors->ElementAt(i))[2];
    gradientIndex++;
  }

  header_struct hdr;
  hdr.gradients = gradients;
  hdr.numberOfBaselineImages = numberOfImages - numberOfGradientImages;
  return hdr;
}

void UpdateMetaDataDictionary(itk::MetaDataDictionary &new_dico, itk::MetaDataDictionary &dico, vnl_matrix<double> new_gradients, unsigned int numBaselines)
{
  std::vector<std::string> imgMetaKeys = dico.GetKeys();
  std::vector<std::string>::const_iterator itKey = imgMetaKeys.begin();
  std::string metaString;
  for (; itKey != imgMetaKeys.end(); itKey ++)
  {
    itk::ExposeMetaData<std::string> (dico, *itKey, metaString);
    if (itKey->find("DWMRI_gradient") == std::string::npos)
      itk::EncapsulateMetaData<std::string>(new_dico, *itKey, metaString);
  }
   
  std::string key_string("DWMRI_gradient_%04i");
  std::string value_string("%lf %lf %lf");
  char key[50];
  char value[100];
  for (unsigned int i=0; i < numBaselines; i++)
  {
    sprintf(key , key_string.c_str(), i);
    sprintf(value , value_string.c_str(), 0.0, 0.0, 0.0);
    itk::EncapsulateMetaData<std::string>(new_dico, key, value);
  }
  for (unsigned int i=numBaselines; i < new_gradients.rows() + numBaselines; i++)
  {
    sprintf(key , key_string.c_str(), i);
    sprintf(value , value_string.c_str(), new_gradients[i-numBaselines][0], new_gradients[i-numBaselines][1], new_gradients[i-numBaselines][2]);
    itk::EncapsulateMetaData<std::string>(new_dico, key, value);
  }

}

void PrintVertices(vnl_matrix<double> vertices)
{
  for (unsigned int i = 0; i < vertices.rows(); i ++)
  {
    std::cout << vertices(i, 0) << " " << vertices(i, 1) << " " << vertices(i, 2) << std::endl;
  }
}

class RotateFunctor
{
  public:
    RotateFunctor() {}
    ~RotateFunctor() {}

    void PrintMatrixRow(vnl_matrix<double> matrix, int row) const
    {
      //std::cout << "matrix is " << matrix.rows() << " by " << matrix.columns() << std::endl;
      for (unsigned int i = 0; i < matrix.cols(); i ++)
      {
        std::cout << matrix(row, i) << " ";
      }
      std::cout  << std::endl;
      cout.flush();
    }

    bool operator!=( const RotateFunctor & ) const
    {
      return false;
    }
    bool operator==( const RotateFunctor & other ) const
    {
      return !(*this != other);
    }

    inline vnl_matrix<double> operator() (const vnl_matrix<double> & gradient_matrix, const vnl_matrix<double> & rotation_matrix) const
    {
      //this->PrintMatrixRow(gradient_matrix, 0);
      //this->PrintMatrixRow(gradient_matrix, 1);
      //this->PrintMatrixRow(gradient_matrix, 2);
      //std::cout << "--->" << std::endl;
      //this->PrintMatrixRow(gradient_matrix*rotation_matrix, 0);
      //this->PrintMatrixRow(gradient_matrix*rotation_matrix, 1);
      //this->PrintMatrixRow(gradient_matrix*rotation_matrix, 2);
      //return gradient_matrix * rotation_matrix;
      vnl_matrix<double> new_gradients = gradient_matrix * rotation_matrix;
      /* normalize rows */
      typedef vnl_numeric_traits<double>::abs_t abs_t;
      for (unsigned int i=0; i<new_gradients.rows(); ++i) 
      {
        //double rn = new_gradients.get_row(i).rms();
        double rn = sqrt(new_gradients(i,0)*new_gradients(i,0) + new_gradients(i,1)*new_gradients(i,1) + new_gradients(i,2)*new_gradients(i,2));
        if (rn > 0) 
        {
          for (unsigned int j=0; j<new_gradients.cols(); ++j)
            new_gradients(i,j) = new_gradients(i,j)/rn;
        }
        //assert( fabs(new_gradients.get_row(i).rms() - 1.0) < itk::NumericTraits<double>::epsilon() );
        //assert( fabs(new_gradients(i,0)) < 1.0 && fabs(new_gradients(i,1)) < 1.0 && fabs(new_gradients(i,2)) < 1.0 );
        if( fabs(new_gradients(i,0)) > 1.0 || fabs(new_gradients(i,1)) > 1.0 || fabs(new_gradients(i,2)) > 1.0 )
        {
          std::cout << "rn: " << rn << ", rms: " << new_gradients.get_row(i).rms() << ", " << new_gradients.get_row(i) << std::endl; 
          cout.flush();
        }

        //if( fabs(new_gradients.get_row(i).rms() - 1.0) > 3*itk::NumericTraits<double>::epsilon() )
        //{
          //std::cout << "size of gradient row is " << new_gradients.get_row(i).rms()  << std::endl;
          //cout.flush();
        //}

      }
      return new_gradients;
    }
};

template< class PixelType > 
unsigned int ComputeSH( parameters args )
{
  const unsigned int Dimension = 3;
  typedef itk::VectorImage< PixelType , Dimension > VectorImageType;
  typedef itk::ImageFileReader< VectorImageType >   ImageReaderType;
  typedef itk::ImageFileWriter< VectorImageType >   WriterType;
  itk::MetaDataDictionary input_dico;
  itk::MetaDataDictionary output_dico;

  typedef vnl_matrix<double> MatrixType;  //hardcoded the type for now because some vnl implementations are limited to just a few (including 'double')
  typedef vnl_vector<double> VectorType;  //hardcoded the type for now because some vnl implementations are limited to just a few (including 'double')

  /* Read the DWI image to be resampled */
  typename ImageReaderType::Pointer imageReader = ImageReaderType::New();
  imageReader->SetFileName( args.input_image.c_str() );
  imageReader->Update();

  /* Get the gradients and number of baseline images */
  input_dico = imageReader->GetOutput()->GetMetaDataDictionary(); //save metadata dictionary
  PrintDictionary(input_dico);
  header_struct hdr = GetGradients(input_dico);
  MatrixType gradients = hdr.gradients; 
  unsigned int numberOfBaselineImages = hdr.numberOfBaselineImages;
  MatrixType duplicated_gradients(2*gradients.rows(), 3);
  duplicated_gradients.update(gradients, 0, 0);
  duplicated_gradients.update(gradients * -1, gradients.rows(), 0);

  /* Make a volume of gradients */
  typedef itk::Image<MatrixType, 3> MatrixImageType;
  typename MatrixImageType::RegionType region;
  typename MatrixImageType::IndexType start_index;
  start_index.Fill(0);
  region.SetIndex(start_index);
  region.SetSize(imageReader->GetOutput()->GetLargestPossibleRegion().GetSize());
  typename MatrixImageType::Pointer gradient_image = MatrixImageType::New();
  gradient_image->SetRegions(region);
  gradient_image->Allocate();
  gradient_image->FillBuffer(duplicated_gradients);

  /* read in rotation matrix */
 /* MATFile *mfile = matOpen("/spl_unsupported/pnlfs/reckbo/projects/CreateDWIAtlas/tests/input/01019-Rgd-fa-Rotation.mat", "r");*/
  //mxArray *rotations = matGetVariable(mfile, "R");
  //mxArray *dJ = matGetVariable(mfile, "dJ"); 
  ////mwSize num_new_dims = 3;
  ////mwSize new_dims[] = {3, 3, 1762560};
  ////mxSetDimensions(rotations, new_dims, num_new_dims);
  //double *rot = mxGetPr(rotations);
  //double *dJ_ptr = mxGetPr(dJ);
  //typename MatrixImageType::RegionType region2;
  //typename MatrixImageType::IndexType start_index2;
  //start_index2.Fill(0);
  //region2.SetIndex(start_index2);
  //region2.SetSize(imageReader->GetOutput()->GetLargestPossibleRegion().GetSize());
  //typename MatrixImageType::Pointer rotation_image = MatrixImageType::New();
  //rotation_image->SetRegions(region2);
  //rotation_image->Allocate();
  //typename itk::ImageRegionIterator< MatrixImageType > in( rotation_image,  rotation_image->GetLargestPossibleRegion() );
  //MatrixType R(3,3);
  //mwIndex subs[] = {0, 0, 0, 0, 0};
  //mwIndex index;
  //mwSize num_dims = 5;
  //for( in.GoToBegin(); !in.IsAtEnd(); ++in )
  //{
    //typename MatrixImageType::IndexType idx = in.GetIndex();
    //for (int i = 0; i < 3; i++)
    //{
      //for (int j = 0; j < 3; j++)
      //{
        //subs[0] = i;
        //subs[1] = j;
        //subs[2] = idx[0];
        //subs[3] = idx[1];
        //subs[4] = idx[2];
        //index = mxCalcSingleSubscript(rotations, num_dims, subs);
        //R(i,j) = rot[index];
      //}
    //}
    //if (vnl_determinant<double>(R) < 0 || dJ_ptr[index] < 0.01)
    //{
      //R.set_identity();
    //}

    //in.Set(R);
    ////PrintMatrixRow(R, 0);
    ////PrintMatrixRow(R, 1);
    ////PrintMatrixRow(R, 2);
    ////std::cout << "---" << std::endl;
    ////cout.flush();
 /* }*/

  /* Compute rotations */
  typedef itk::Vector<float, Dimension>  VectorPixelType;
  //typedef itk::Image< PixelType, Dimension >  ImageType;
  typedef itk::OrientedImage< PixelType , Dimension > ImageType;
  typedef itk::Image<VectorPixelType, Dimension>  DeformationFieldType;
  typedef itk::ExtractRotationsFilter< DeformationFieldType, double> ExtractRotationsFilterType; 
  typedef itk::ImageFileReader< DeformationFieldType >    DeformationReaderType;
  DeformationReaderType::Pointer  fieldReader = DeformationReaderType::New();
  fieldReader->SetFileName( args.warp.c_str() );
  fieldReader->Update();
  ExtractRotationsFilterType::Pointer rotations_filter = ExtractRotationsFilterType::New();
  rotations_filter->SetInput(fieldReader->GetOutput());
  //rotations_filter->Update();
  rotations_filter->UpdateLargestPossibleRegion();

  typedef itk::BinaryFunctorImageFilter< MatrixImageType, MatrixImageType, MatrixImageType, RotateFunctor > RotateFilterType;
  typename RotateFilterType::Pointer rotate_filter = RotateFilterType::New();
  rotate_filter->SetInput1(gradient_image);
  //rotate_filter->SetInput2(rotation_image);
  rotate_filter->SetInput2(rotations_filter->GetOutput());
  /*debug*/
  //rotate_filter->Update();
  //exit(1);
  /*debug*/

  /* Get new sample directions */
  MatrixType vertices = args.resample_self ? gradients : sample_sphere_as_icosahedron(2);

  /* Update output DWI header with new sample directions */
  if (args.without_baselines)
    UpdateMetaDataDictionary(output_dico, input_dico, vertices, 0);
  else
    UpdateMetaDataDictionary(output_dico, input_dico, vertices, numberOfBaselineImages);

  /* Create the SH filter */
  typedef itk::SHFilter< VectorImageType > FilterType;
  typename FilterType::Pointer filter = FilterType::New();
  filter->SetSamples(vertices);
  filter->SetInput1(imageReader->GetOutput());
  filter->SetInput2(rotate_filter->GetOutput());
  if (args.without_baselines)
  {
    filter->GetFunctor().ExcludeBaselineImagesFromOutput();
    filter->SetOutputLength(vertices.rows());
  }
  else
  {
    filter->GetFunctor().IncludeBaselineImagesInOutput();
    filter->SetOutputLength(vertices.rows() + numberOfBaselineImages);
  }
  filter->GetOutput()->SetMetaDataDictionary(output_dico);
  typedef itk::ImageFileWriter< VectorImageType >   WriterType2;
  typename WriterType2::Pointer  writer2 =  WriterType2::New();
  writer2->SetFileName( args.output_image.c_str() );
  writer2->SetInput( filter->GetOutput() );
  writer2->SetUseCompression( true );
  try
  {
    writer2->Update();
  }
  catch( itk::ExceptionObject& err )
  {
    std::cout << "Could not write SH coefficients" << std::endl;
    std::cout << err << std::endl;
    exit( EXIT_FAILURE );
  }

  return EXIT_SUCCESS;

}

template< class PixelType > 
int Warp( parameters &args )
{
  //if (args.resample || args.resample_self)
  if ( args.warp.empty() && (args.resample || args.resample_self) ) 
  {
    return ComputeSH<PixelType>(args);
  }

  const unsigned int Dimension = 3;
  typedef itk::Vector<float, Dimension>  VectorPixelType;
  //typedef itk::Image< PixelType, Dimension >  ImageType;
  typedef itk::OrientedImage< PixelType , Dimension > ImageType;
  typedef itk::Image<VectorPixelType, Dimension>  DeformationFieldType;
  typedef itk::WarpImageFilter <ImageType, ImageType, DeformationFieldType>  WarperType;
  typedef itk::ImageFileReader< DeformationFieldType >    DeformationReaderType;
  typedef itk::VectorImage< PixelType , Dimension > VectorImageType;
  typedef itk::ImageFileWriter< VectorImageType >   WriterType;
  //typedef itk::ImageFileWriter< ImageType >   WriterType;
  //itk::MetaDataDictionary dico;

  /* read in input image */
  typedef itk::ImageFileReader< VectorImageType >   ImageReaderType;
  typename ImageReaderType::Pointer imageReader = ImageReaderType::New();
  imageReader->SetFileName( args.input_image.c_str() );
  imageReader->Update();
  itk::MetaDataDictionary input_dico = imageReader->GetOutput()->GetMetaDataDictionary();

  /* read in deformation field */
  DeformationReaderType::Pointer  fieldReader = DeformationReaderType::New();
  fieldReader->SetFileName( args.warp.c_str() );
  fieldReader->Update();

  /* separate into a vector */
  std::vector< typename ImageType::Pointer > vectorImage;
  std::vector< typename ImageType::Pointer > vectorOutputImage ;
  SeparateImages< PixelType >( imageReader->GetOutput() , vectorImage ) ;

  /* debug */
  /* If DWI, resample the gradient directions */
  //if (vectorImage.size() > 1)
  //{
    //std::cout << "Resampling the warped DWI" << std::endl;
    //return ComputeSH<PixelType>(args);
  //}
  /* debug */

  /* warp the image */
  typename WarperType::Pointer   warper = WarperType::New();
  warper->SetDeformationField( fieldReader->GetOutput() );
  for( ::size_t i = 0; i < vectorImage.size(); i++ )
  {
    std::cout << "number of components: " << vectorImage.size() << ", iteration: " << i <<  std::endl;
    warper->SetInput( vectorImage[i] );
    warper->SetOutputSpacing( vectorImage[i]->GetSpacing() );
    warper->SetOutputOrigin( vectorImage[i]->GetOrigin() );
    warper->SetOutputDirection( vectorImage[i]->GetDirection() );
    warper->SetOutputSize( vectorImage[i]->GetLargestPossibleRegion().GetSize() );
    warper->Update();
    vectorOutputImage.push_back( warper->GetOutput() );
    vectorOutputImage[i]->DisconnectPipeline();
  }

  typename itk::VectorImage< PixelType, 3 >::Pointer outputImage = itk::VectorImage< PixelType , 3 >::New() ;
  AddImage< PixelType >( outputImage , vectorOutputImage ) ;
  vectorOutputImage.clear() ;
  outputImage->SetMetaDataDictionary(input_dico);

  //warper->SetInput( imageReader->GetOutput() );
  //warper->SetOutputSpacing( imageReader->GetOutput()->GetSpacing() );
  //warper->SetOutputOrigin( imageReader->GetOutput()->GetOrigin() );
  //warper->SetOutputDirection( imageReader->GetOutput()->GetDirection() );

  typename WriterType::Pointer  writer =  WriterType::New();
  writer->SetFileName( args.output_image );
  writer->SetInput( outputImage );
  writer->SetUseCompression( true );
  try
  {
    writer->Update();
  }
  catch( itk::ExceptionObject& err )
  {
    std::cout << "Could not write warped image" << std::endl;
    std::cout << err << std::endl;
    exit( EXIT_FAILURE );
  }

  /* If DWI, resample the gradient directions */
  if (vectorImage.size() > 1)
  {
    std::cout << "Resampling the warped DWI" << std::endl;
    args.input_image = args.output_image;
    return ComputeSH<PixelType>(args);
  }

  return EXIT_SUCCESS;
}

}

int main( int argc, char * argv[] )
{
  PARSE_ARGS;
  parameters args;
  args.warp = warp;
  args.output_image = output_image;
  args.input_image = input_image;
  args.resample = resample;
  args.resample_self = resample_self;
  args.without_baselines = without_baselines;

  itk::ImageIOBase::IOPixelType pixelType;
  itk::ImageIOBase::IOComponentType componentType;
  GetImageType( args.input_image , pixelType , componentType );


  switch( componentType )
   {
      case itk::ImageIOBase::UCHAR:
         return Warp< unsigned char >( args );
         break;
      case itk::ImageIOBase::CHAR:
         return Warp< char >( args );
         break;
      case itk::ImageIOBase::USHORT:
         return Warp< unsigned short >( args );
         break;
      case itk::ImageIOBase::SHORT:
         return Warp< short >( args );
         break;
      case itk::ImageIOBase::UINT:
         return Warp< unsigned int >( args );
         break;
      case itk::ImageIOBase::INT:
         return Warp< int >( args );
         break;
      case itk::ImageIOBase::ULONG:
         return Warp< unsigned long >( args );
         break;
      case itk::ImageIOBase::LONG:
         return Warp< long >( args );
         break;
      case itk::ImageIOBase::FLOAT:
         return Warp< float >( args );
         break;
      case itk::ImageIOBase::DOUBLE:
         return Warp< double >( args );
         break;
      case itk::ImageIOBase::UNKNOWNCOMPONENTTYPE:
      default:
         std::cerr << "Unknown component type" << std::endl;
         break;
   }
   return EXIT_FAILURE;
}
