#include "WarpVolumeCLP.h"
#include <itkImage.h>
#include <itkOrientedImage.h>
#include <itkImageIOBase.h>
#include <itkImageFileReader.h>
#include <itkImageRegionIterator.h>
#include <itkVectorImage.h>
#include <itkVariableLengthVector.h>
#include <vector>
#include <algorithm>
#include "itkWarpImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkMultivariateLegendrePolynomial.h"
#include "SphericalHarmonicPolynomial.h"
#include <math.h>

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
  std::string inputVolume;
  std::string outputVolume;
  std::string warp;
  bool resample;
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

       if (std::find(vertices.begin(), vertices.end(), v1) != vertices.end() == false)
         vertices.push_back(v1);

       if (std::find(vertices.begin(), vertices.end(), v2) != vertices.end() == false)
         vertices.push_back(v2);

       if (std::find(vertices.begin(), vertices.end(), v3) != vertices.end() == false)
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

//std::string WarpedImageName(std::string outputDir, std::string filename)
//{
  //std::stringstream result;
  //result << outputDir << "/" << itksys::SystemTools::GetFilenameWithoutExtension(filename) << "_warped.nrrd";
  //return result.str();
//}

template< class PixelType>
vnl_matrix<double> GetSHBasis3(vnl_matrix<double> samples, int L)
{
  int numcoeff = (L+1)*(L+2)/2;
  typedef vnl_matrix<double> MatrixType;
  MatrixType Y(samples.rows(), numcoeff);

  /* this is the makespharms(u, L) function in Yogesh's Matlab code (/home/yogesh/yogesh_pi/phd/dwmri/fODF_SH/makespharms.m) */
  typedef neurolib::SphericalHarmonicPolynomial<3> SphericalHarmonicPolynomialType;
  SphericalHarmonicPolynomialType *sphm = new SphericalHarmonicPolynomialType();
  for (int i = 0; i < samples.rows(); i++)
  {
    double theta = acos( samples(i,2) );
    double varphi = atan2( samples(i,1), samples(i,0) );
    if (varphi < 0) 
      varphi = varphi + 2*M_PI;
    int coeff_i = 0;
    Y(i,coeff_i) = sphm->SH(0,0,theta,varphi);
    coeff_i++;
    //std::cout << sphm->SH(0,0,theta,varphi) << " ";
    for (int l = 2; l <=L; l+=2)
    {
      for (int m = l; abs(m) <= l; m--)
      {
        Y(i,coeff_i) = sphm->SH(l,m,theta,varphi);
        coeff_i++;
      }
    }
  }
  std::cout << "num rows of Y is: " << Y.rows() << std::endl;
  return Y;
}

template< class PixelType > 
void PrintMatrix(vnl_matrix<double> matrix, int col)
{
  std::cout << "matrix is " << matrix.rows() << " by " << matrix.columns() << std::endl;
  for (unsigned int i = 0; i < matrix.rows(); i ++)
  {
    std::cout << matrix(i, col) << std::endl;
  }
  std::cout  << std::endl;
}

template< class PixelType > 
void PrintVector(vnl_vector<double> vector)
{
  for (unsigned int i = 0; i < vector.size(); i ++)
  {
    std::cout << vector(i) << std::endl;
  }
  std::cout  << std::endl;
}

template< class PixelType > 
unsigned int ComputeSH( parameters args )
{
  const unsigned int Dimension = 3;
  typedef itk::VectorImage< PixelType , Dimension > VectorImageType;
  typedef itk::ImageFileReader< VectorImageType >   ImageReaderType;
  typedef itk::ImageFileWriter< VectorImageType >   WriterType;

  typedef vnl_matrix<double> MatrixType;  //hardcoded the type for now because some vnl implementations are limited to just a few (including 'double')
  typedef vnl_vector<double> VectorType;  //hardcoded the type for now because some vnl implementations are limited to just a few (including 'double')
  const unsigned int L = 8;
  const unsigned int num_basis_functions = (L+1)*(L+2)/2;

  /* Get the SH basis function values for our new set of sphere samples */
  MatrixType vertices = sample_sphere_as_icosahedron(2); //std::cout << "num vertices is " << vertices.rows() << std::endl;
  MatrixType newY = GetSHBasis3<double>(vertices, L); //std::cout << "newY size is: " << newY.rows() << std::endl;

  /* Read in the gradients from the DWI and put into vtkDoubleArray 'grads' */
  vtkSmartPointer<vtkNRRDReader> reader = vtkNRRDReader::New();
  reader->SetFileName(args.inputVolume.c_str());
  reader->Update();
  vtkSmartPointer<vtkDoubleArray> bValues = vtkDoubleArray::New();
  vtkSmartPointer<vtkDoubleArray> grads = vtkDoubleArray::New();
  vtkSmartPointer<vtkMRMLNRRDStorageNode> helper = vtkMRMLNRRDStorageNode::New();
  if ( !helper->ParseDiffusionInformation(reader,grads,bValues) )
    {
    std::cerr << "Error parsing Diffusion information" << std::endl;
    return EXIT_FAILURE;
    }

  /* read in rotation matrix */
  //MATFile *mfile = matOpen("/spl_unsupported/pnlfs/reckbo/projects/CreateDWIAtlas/tests/input/01019-Rgd-fa-Rotation.mat", "r");
  //mxArray *rotations = matGetVariable(mfile, "R");
  //double *rot = mxGetPr(rotations);
  
  /* Transfer the gradients to a vnl matrix, because we're going use vnl math operations on it */
  MatrixType gradients( 2*(grads->GetNumberOfTuples()-8), 3 );
  for (int k = 0; k < 2; k++)
    for (int i = 8; i < grads->GetNumberOfTuples(); i++)
      for (int j = 0; j < 3; j++)
      {
        if (k == 0)
          gradients(i-8,j) = grads->GetComponent(i,j);
        else
        {
          gradients(i-8+grads->GetNumberOfTuples()-8,j) = -1 * grads->GetComponent(i,j);
        }
      }

  
  /* Read the DWI image to be resampled */
  typename ImageReaderType::Pointer imageReader = ImageReaderType::New();
  imageReader->SetFileName( args.inputVolume.c_str() );
  imageReader->Update();

  /* Configure the new SH image */
  typename VectorImageType::Pointer outputImage = VectorImageType::New();
  outputImage->SetRegions( imageReader->GetOutput()->GetLargestPossibleRegion().GetSize() );
  outputImage->SetOrigin( imageReader->GetOutput()->GetOrigin() );
  outputImage->SetDirection( imageReader->GetOutput()->GetDirection() );
  outputImage->SetSpacing( imageReader->GetOutput()->GetSpacing() );
  //outputImage->SetVectorLength( imageReader->GetOutput()->GetVectorLength() );
  outputImage->SetVectorLength( 162 ); //TODO: soft code this
  outputImage->Allocate();

  //int isNotZero = 0;
  //MatrixType R(3,3);
  //mwIndex subs[] = {0, 0, 0, 0, 0};
  //mwIndex index;
  //int count = 0;

  /* Compute B, a constant vector with size num_basis_functions. We're going to use this vector when computing the SH projection of the gradient directions below */
  VectorType r(num_basis_functions );
  VectorType a(1);
  a(0) = 1;
  int end = 0;
  for (int l = 0; l <= L; l+=2)
  {
    a.set_size(2*l+1);
    a.fill(l);
    r.update(a, end); end += a.size(); }
  VectorType B = element_product(r, r+1);
  B = element_product(B,B);

  /* Get the value for each of the (L+1)*(L+2)/2 SH basis functions at each gradient direction */
  MatrixType Y2 = GetSHBasis3<double>(gradients, L);

  /* Perform part of the gradient SH projection computation */ 
  MatrixType Y2_t = Y2.transpose();
  MatrixType denominator = Y2_t * Y2;
  vnl_diag_matrix<double> diag =  vnl_diag_matrix<double>(0.003 * B);
  denominator = denominator +  diag;
  denominator = vnl_matrix_inverse<double>( denominator );

  /* do for each voxel */
  typename itk::ImageRegionIterator< VectorImageType > in( imageReader->GetOutput(),  imageReader->GetOutput()->GetLargestPossibleRegion() );
  for( in.GoToBegin(); !in.IsAtEnd(); ++in )
  {
    /* get S = [data(j,:) data(j,:)], the data vector with twice the size as the number of gradients */
    typename VectorImageType::IndexType idx = in.GetIndex();
    itk::VariableLengthVector< double > data = in.Get(); //itk::VariableLengthVector< PixelType > data = in.Get();
    VectorType S(2*data.GetNumberOfElements()-16); //TODO: the baseline slices are hardcoded to be 8
    for (unsigned int i = 8; i < data.GetNumberOfElements(); i++) 
    {
      S(i-8) = data.GetElement(i);
      S(i-8+data.GetNumberOfElements()-8) = data.GetElement(i);
    }

    /* Compute the SH projection, 'Cs', of this voxel's gradient function onto (L+1)(L+2)/2 basis functions. So 'Cs' is a vector of size (L+1)(L+2)/2. */
    VectorType Cs = denominator * Y2_t * S;

    /* Compute the voxel's values at the new sample directions */
    VectorType sh_coef = newY * Cs;

    /* Save the new values to the output image */ 
    itk::VariableLengthVector<double> sh_coef_final;
    sh_coef_final.SetSize(sh_coef.size());
    for (int i = 0; i < sh_coef.size(); i ++)
      sh_coef_final[i] = sh_coef[i];
    outputImage->SetPixel(idx, sh_coef_final);
  }

  typename WriterType::Pointer  writer =  WriterType::New();
  writer->SetFileName( args.outputVolume.c_str() );
  writer->SetInput( outputImage );
  writer->SetUseCompression( true );
  try
  {
    writer->Update();
  }
  catch( itk::ExceptionObject& err )
  {
    std::cout << "Could not write SH coefficients" << std::endl;
    std::cout << err << std::endl;
    exit( EXIT_FAILURE );
  }

  return 1;
}

template< class PixelType > 
int Warp( parameters &args )
{
  if (args.resample)
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

  DeformationReaderType::Pointer  fieldReader = DeformationReaderType::New();
  fieldReader->SetFileName( args.warp.c_str() );
  fieldReader->Update();

  /* separate into a vector */
  typedef itk::ImageFileReader< VectorImageType >   ImageReaderType;
  typename ImageReaderType::Pointer imageReader = ImageReaderType::New();
  imageReader->SetFileName( args.inputVolume.c_str() );
  imageReader->Update();

  /* warp the image(s) */
  typename WarperType::Pointer   warper = WarperType::New();
  warper->SetDeformationField( fieldReader->GetOutput() );

  std::vector< typename ImageType::Pointer > vectorOutputImage ;
  std::vector< typename ImageType::Pointer > vectorOfImage;
  SeparateImages< PixelType >( imageReader->GetOutput() , vectorOfImage ) ;

  for( ::size_t i = 0; i < vectorOfImage.size(); i++ )
  {
    std::cout << "number of components: " << vectorOfImage.size() << ", iteration: " << i <<  std::endl;
    warper->SetInput( vectorOfImage[i] );
    warper->SetOutputSpacing( vectorOfImage[i]->GetSpacing() );
    warper->SetOutputOrigin( vectorOfImage[i]->GetOrigin() );
    warper->SetOutputDirection( vectorOfImage[i]->GetDirection() );
    warper->SetOutputSize( vectorOfImage[i]->GetLargestPossibleRegion().GetSize() );
    warper->Update();
    vectorOutputImage.push_back( warper->GetOutput() );
    vectorOutputImage[i]->DisconnectPipeline();
  }

  typename itk::VectorImage< PixelType, 3 >::Pointer outputImage = itk::VectorImage< PixelType , 3 >::New() ;
  AddImage< PixelType >( outputImage , vectorOutputImage ) ;
  vectorOutputImage.clear() ;

  //warper->SetInput( imageReader->GetOutput() );
  //warper->SetOutputSpacing( imageReader->GetOutput()->GetSpacing() );
  //warper->SetOutputOrigin( imageReader->GetOutput()->GetOrigin() );
  //warper->SetOutputDirection( imageReader->GetOutput()->GetDirection() );

  typename WriterType::Pointer  writer =  WriterType::New();
  //writer->SetFileName( WarpedImageName(args.resultsDirectory, args.inputVolume) );
  writer->SetFileName( args.outputVolume );
  writer->SetInput( outputImage );
  //writer->SetInput( imageReader->GetOutput() );
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

  /* debug */
  //writer->SetInput( vectorOfImage[7] );
  //writer->SetFileName( "./component7.nrrd" );
  //writer->SetUseCompression( true );
  //try
  //{
    //writer->Update();
  //}
  //catch( itk::ExceptionObject& err )
  //{
    //std::cout << "Could not write 5th component" << std::endl;
    //std::cout << err << std::endl;
    //exit( EXIT_FAILURE );
  //}

}

}

int main( int argc, char * argv[] )
{
  PARSE_ARGS;
  parameters args;
  args.warp = warp;
  args.outputVolume = outputVolume;
  args.inputVolume = inputVolume;
  args.resample = resample;

  std::cout << "warp:" << args.warp << std::endl;
  std::cout << "input volume:" << args.inputVolume << std::endl;
  std::cout << "output volume:" << args.outputVolume << std::endl;
  std::cout << "resample:" << args.resample << std::endl;

  itk::ImageIOBase::IOPixelType pixelType;
  itk::ImageIOBase::IOComponentType componentType;
  GetImageType( args.inputVolume , pixelType , componentType );


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
