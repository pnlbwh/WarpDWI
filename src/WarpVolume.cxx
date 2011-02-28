#include "WarpVolumeCLP.h"
#include <itkImage.h>
#include <itkOrientedImage.h>
#include <itkImageIOBase.h>
#include <itkImageFileReader.h>
#include <itkImageRegionIterator.h>
#include <itkVectorImage.h>
#include <itkVariableLengthVector.h>
#include <vector>
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


struct parameters
{
  std::string inputVolume;
  std::string warp;
  std::string resultsDirectory;
};



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

std::string WarpedImageName(std::string outputDir, std::string filename)
{
  std::stringstream result;
  result << outputDir << "/" << itksys::SystemTools::GetFilenameWithoutExtension(filename) << "_warped.nrrd";
  return result.str();
}

//void GetImageType( std::string fileName ,
                   //itk::ImageIOBase::IOPixelType &pixelType ,
                   //itk::ImageIOBase::IOComponentType &componentType
                 //)
//{
   //typedef itk::Image< unsigned char , 3 > ImageType;
   //itk::ImageFileReader< ImageType >::Pointer imageReader;
   //imageReader = itk::ImageFileReader< ImageType >::New();
   //imageReader->SetFileName( fileName.c_str() );
   //imageReader->UpdateOutputInformation();
   //pixelType = imageReader->GetImageIO()->GetPixelType();
   //componentType = imageReader->GetImageIO()->GetComponentType();
//}


//template< class PixelType>
//void GetSHBasis2(vnl_matrix<double> &Y, vtkSmartPointer<vtkDoubleArray> &grads, int L)
template< class PixelType>
vnl_matrix<double> GetSHBasis2(vtkSmartPointer<vtkDoubleArray> &grads, int L)
{
  int numcoeff = (L+1)*(L+2)/2;
  //typedef vnl_matrix<PixelType> MatrixType;
  typedef vnl_matrix<double> MatrixType;
  MatrixType Y(2*(grads->GetNumberOfTuples()-8), numcoeff);
  std::cout << "Y is " << Y.rows() << " by " << Y.columns() << std::endl;
  //Y->SetNumberOfComponents( numcoeff );
  /* makespharms(u, L) */
  typedef neurolib::SphericalHarmonicPolynomial<3> SphericalHarmonicPolynomialType;
  SphericalHarmonicPolynomialType *sphm = new SphericalHarmonicPolynomialType();
  int flag = 1;
  int offset = -8;
  for (int j = 0; j < 2; j++)
  {
    //for (int i = 8; i < grads->GetNumberOfTuples(); i ++)
    for (int i = 8; i < grads->GetNumberOfTuples(); i++)
    {
      double theta = acos(flag*grads->GetComponent(i,2));
      double varphi = atan2(flag*grads->GetComponent(i,1), flag*grads->GetComponent(i,0) );
      if (varphi < 0) varphi = varphi + 2*M_PI;
      //double coeff[numcoeff];  
      int coeff_i = 0;
      //coeff[coeff_i] = sphm->SH(0,0,theta,varphi);
      Y(i+offset,coeff_i) = sphm->SH(0,0,theta,varphi);
      coeff_i++;
      //std::cout << sphm->SH(0,0,theta,varphi) << " ";
      for (int l = 2; l <=L; l+=2)
      {
        for (int m = l; abs(m) <= l; m--)
        {
          //std::cout << sphm->SH(l,m,theta,varphi) << " ";
          //coeff[coeff_i] = sphm->SH(l,m,theta,varphi);
          Y(i+offset,coeff_i) = sphm->SH(l,m,theta,varphi);
          coeff_i++;
        }
      }
      //Y->InsertNextTuple(coeff);
    }
    flag = -1;
    offset = offset + grads->GetNumberOfTuples() - 8;
  }
  return Y;
}

template< class PixelType>
vnl_matrix<double> GetSHBasis3(vnl_matrix<double> gradients, int L)
{
  int numcoeff = (L+1)*(L+2)/2;
  //typedef vnl_matrix<PixelType> MatrixType;
  typedef vnl_matrix<double> MatrixType;
  MatrixType Y(gradients.rows()*2, numcoeff);

  /* makespharms(u, L) */
  typedef neurolib::SphericalHarmonicPolynomial<3> SphericalHarmonicPolynomialType;
  SphericalHarmonicPolynomialType *sphm = new SphericalHarmonicPolynomialType();
  int flag = 1;
  int offset = 0;
  for (int j = 0; j < 2; j++)
  {
    for (int i = 0; i < gradients.rows(); i++)
    {
      double theta = acos( flag*gradients(i,2) );
      double varphi = atan2( flag*gradients(i,1), flag*gradients(i,0) );
      if (varphi < 0) varphi = varphi + 2*M_PI;
      //double coeff[numcoeff];  
      int coeff_i = 0;
      Y(i+offset,coeff_i) = sphm->SH(0,0,theta,varphi);
      coeff_i++;
      //std::cout << sphm->SH(0,0,theta,varphi) << " ";
      for (int l = 2; l <=L; l+=2)
      {
        for (int m = l; abs(m) <= l; m--)
        {
          //std::cout << sphm->SH(l,m,theta,varphi) << " ";
          //coeff[coeff_i] = sphm->SH(l,m,theta,varphi);
          Y(i+offset,coeff_i) = sphm->SH(l,m,theta,varphi);
          coeff_i++;
        }
      }
      //Y->InsertNextTuple(coeff);
    }
    flag = -1;
    offset = gradients.rows();
  }
  return Y;
}

template< class PixelType>
void GetSHBasis(vtkSmartPointer<vtkDoubleArray> &Y, vtkSmartPointer<vtkDoubleArray> &grads, int L)
{
  int numcoeff = (L+1)*(L+2)/2;
  Y->SetNumberOfComponents( numcoeff );
  /* makespharms(u, L) */
  typedef neurolib::SphericalHarmonicPolynomial<3> SphericalHarmonicPolynomialType;
  SphericalHarmonicPolynomialType *sphm = new SphericalHarmonicPolynomialType();
  int flag = 1;
  for (int j = 0; j < 2; j++)
  {
    for (int i = 8; i < grads->GetNumberOfTuples(); i ++)
    {
      double theta = acos(flag*grads->GetComponent(i,2));
      double varphi = atan2(flag*grads->GetComponent(i,1), flag*grads->GetComponent(i,0) );
      if (varphi < 0) varphi = varphi + 2*M_PI;
      double coeff[numcoeff];  
      int coeff_i = 0;
      coeff[coeff_i] = sphm->SH(0,0,theta,varphi);
      coeff_i++;
      //std::cout << sphm->SH(0,0,theta,varphi) << " ";
      for (int l = 2; l <=L; l+=2)
      {
        for (int m = l; abs(m) <= l; m--)
        {
          //std::cout << sphm->SH(l,m,theta,varphi) << " ";
          coeff[coeff_i] = sphm->SH(l,m,theta,varphi);
          coeff_i++;
        }
      }
      Y->InsertNextTuple(coeff);
    }
    flag = -1;
  }
}

template< class PixelType > 
int Warp( parameters &args )
{
  const unsigned int Dimension = 3;
  typedef itk::Vector<float, Dimension>  VectorPixelType;
  //typedef itk::Image< PixelType, Dimension >  ImageType;
  typedef itk::OrientedImage< PixelType , Dimension > ImageType;
  typedef itk::Image<VectorPixelType, Dimension>  DeformationFieldType;
  typedef itk::WarpImageFilter <ImageType, ImageType, DeformationFieldType>  WarperType;
  typedef itk::ImageFileReader< DeformationFieldType >    DeformationReaderType;
  typedef itk::VectorImage< PixelType , Dimension > VectorImageType;
  typedef itk::ImageFileReader< VectorImageType >   ImageReaderType;
  typedef itk::ImageFileWriter< VectorImageType >   WriterType;
  //typedef itk::ImageFileWriter< ImageType >   WriterType;

  //itk::MetaDataDictionary dico;

  DeformationReaderType::Pointer   fieldReader = DeformationReaderType::New();
  fieldReader->SetFileName( args.warp.c_str() );
  fieldReader->Update();

  /* read input volume */
  vtkSmartPointer<vtkNRRDReader> reader = vtkNRRDReader::New();
  reader->SetFileName(args.inputVolume.c_str());
  reader->Update();
  vtkSmartPointer<vtkDoubleArray> bValues = vtkDoubleArray::New();
  vtkSmartPointer<vtkDoubleArray> grads = vtkDoubleArray::New();
  vtkSmartPointer<vtkMRMLNRRDStorageNode> helper = vtkMRMLNRRDStorageNode::New();
  if ( !helper->ParseDiffusionInformation(reader,grads,bValues) )
    {
    //std::cerr << argv[0] << ": Error parsing Diffusion information" << std::endl;
    std::cerr << ": Error parsing Diffusion information" << std::endl;
    return EXIT_FAILURE;
    }

  /* read in rotation matrix */
  MATFile *mfile = matOpen("/spl_unsupported/pnlfs/reckbo/projects/CreateDWIAtlas/tests/input/01019-Rgd-fa-Rotation.mat", "r");
  mxArray *rotations = matGetVariable(mfile, "R");
  mwSize num_new_dims = 3;
  mwSize new_dims[] = {3, 3, 1762560};
  mxSetDimensions(rotations, new_dims, num_new_dims);
  double *rot = mxGetPr(rotations);
  
  mwIndex subs[] = {2, 0, 0};
  mwIndex index = mxCalcSingleSubscript(rotations, num_new_dims, subs);
  //printf("rotation element: %f", rot[index]);
  std::cout << "rotation element: " << rot[index] << std::endl;

  //printf("index: %i", index);


  /* Compute Spherical Harmonic coefficients */
  typedef vnl_matrix<double> MatrixType;
  int L = 8;

  /* Put gradients into a vnl matrix */
  MatrixType gradients(grads->GetNumberOfTuples()-8, 3);
  for (int i = 8; i < grads->GetNumberOfTuples(); i++)
  {
    for (int j = 0; j < 3; j++)
    {
      gradients(i-8,j) = grads->GetComponent(i,j);
    }
  }

  /* Load Rotation */
  MatrixType R(3,3);
  R(0,0) = 0.999450003479549;   
  R(0,1) = 0.017033058532942;   
  R(0,2) = 0.028452863858364;
  R(1,0) = -0.016471001243502;
  R(1,1) = 0.999666831145781;
  R(1,2) = -0.019872916871657;
  R(2,0) = -0.028781880806608;
  R(2,1) = 0.019393339680534;
  R(2,2) = 0.999397569395318;


  //MatrixType Y2(grads->GetNumberOfTuples()-8,numcoeff);
  //MatrixType Y2 = GetSHBasis2<PixelType>(grads, L);
  //std::cout << gradients * R << std::endl;
  MatrixType Y2 = GetSHBasis3<double>(gradients * R, L);
    
  //vtkSmartPointer<vtkDoubleArray> Y = vtkDoubleArray::New();
  //GetSHBasis<PixelType>(Y, grads, L);
  //for (int i = 0; i < Y->GetNumberOfTuples(); i ++)
  //{
      //std::cout << Y->GetComponent(i, 44) << std::endl;
  //}

  /* separate into a vector */
  typename ImageReaderType::Pointer imageReader = ImageReaderType::New();
  imageReader->SetFileName( args.inputVolume.c_str() );
  imageReader->Update();


  /* compute B */
  vnl_vector<double> r(45);
  vnl_vector<double> a(1);
  a(0) = 1;
  int end = 0;
  for (int l = 0; l <= L; l+=2)
  {
    a.set_size(2*l+1);
    a.fill(l);
    r.update(a, end); end += a.size(); }
  vnl_vector<double> B = element_product(r, r+1);
  B = element_product(B,B);



  /* Compute SHestim */
  typename itk::ImageRegionIterator< VectorImageType > in( imageReader->GetOutput(),  imageReader->GetOutput()->GetLargestPossibleRegion() );
  int isNotZero = 0;
  for( in.GoToBegin(); !in.IsAtEnd(); ++in )
  {
    //itk::VariableLengthVector< PixelType > data = in.Get();
    itk::VariableLengthVector< double > data = in.Get();
    vnl_vector<double> S(2*data.GetNumberOfElements()-16);
    for (int i = 8; i < data.GetNumberOfElements(); i++)
    {
      S(i-8) = data.GetElement(i);
      S(i-8+data.GetNumberOfElements()-8) = data.GetElement(i);
      if ( S(i-8) > 0.0 ) isNotZero = 1;
    }

    if (isNotZero)
    {
      std::cout << "Y is " << Y2.rows() << " by " << Y2.columns() << std::endl;
      for (unsigned int i = 0; i < Y2.rows(); i ++)
      {
        std::cout << Y2(i, 44) << std::endl;
      }
      std::cout  << std::endl;

      MatrixType Y2_t = Y2.transpose();
      std::cout << "S is " << S << std::endl;
      vnl_diag_matrix<double> diag =  vnl_diag_matrix<double>(0.003 * B);
      MatrixType denominator = Y2_t * Y2;
      //std::cout << "Y2_t * Y2 is " <<  denominator << std::endl;
      denominator = denominator +  diag;
      //std::cout << "Y2_t * Y2 + 0.003 * diag(B) is " <<  denominator << std::endl;
      denominator = vnl_matrix_inverse<double>( denominator );
      //std::cout << "inverse denominator is " << std::endl <<  denominator << std::endl;
      //std::cout << "denominator is " << denominator.rows() << " by " << denominator.columns() << std::endl;
      vnl_vector<double> numerator =  Y2_t * S;
      //std::cout << "Y2_t * S is " <<  std::endl << Y2_t * S << std::endl;
      std::cout << "result is " <<  denominator * Y2_t * S << std::endl;
      return 1;
    }
  }



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

  typename itk::VectorImage< PixelType, 3 >::Pointer outputImage ;
  outputImage = itk::VectorImage< PixelType , 3 >::New() ;
  AddImage< PixelType >( outputImage , vectorOutputImage ) ;
  vectorOutputImage.clear() ;

  //warper->SetInput( imageReader->GetOutput() );
  //warper->SetOutputSpacing( imageReader->GetOutput()->GetSpacing() );
  //warper->SetOutputOrigin( imageReader->GetOutput()->GetOrigin() );
  //warper->SetOutputDirection( imageReader->GetOutput()->GetDirection() );

  typename WriterType::Pointer  writer =  WriterType::New();
  writer->SetFileName( WarpedImageName(args.resultsDirectory, args.inputVolume) );
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


int main( int argc, char * argv[] )
{
  PARSE_ARGS;

  std::cout << "output directory:" << resultsDirectory << std::endl;
  std::cout << "warp:" << warp << std::endl;
  std::cout << "input volume:" << inputVolume << std::endl;

  itk::ImageIOBase::IOPixelType pixelType;
  itk::ImageIOBase::IOComponentType componentType;
  GetImageType( inputVolume , pixelType , componentType );

  parameters args;
  args.resultsDirectory = resultsDirectory;
  args.warp = warp;
  args.inputVolume = inputVolume;

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
