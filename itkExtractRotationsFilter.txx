/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkDisplacementFieldJacobianDeterminantFilter.txx,v $
  Language:  C++
  Date:      $Date: 2008-12-07 17:07:52 $
  Version:   $Revision: 1.5 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkExtractRotationsFilter_txx
#define __itkExtractRotationsFilter_txx

#include "itkExtractRotationsFilter.h"

#include "itkNeighborhoodAlgorithm.h"
#include "itkImageRegionIterator.h"
#include "itkZeroFluxNeumannBoundaryCondition.h"
#include "itkProgressReporter.h"
#include "itkVectorCastImageFilter.h"

#include "vnl/vnl_math.h"
#include "math.h"

namespace itk
{

template <typename TInputImage, typename TRealType, typename TOutputImage>
ExtractRotationsFilter<TInputImage, TRealType, TOutputImage>
::ExtractRotationsFilter()
{
  m_UseImageSpacing = false;
  m_RequestedNumberOfThreads = this->GetNumberOfThreads();
  for (unsigned int i = 0; i < ImageDimension; i++)
    {
    m_NeighborhoodRadius[i] = 1; // radius of neighborhood we will use
    m_DerivativeWeights[i] = static_cast<TRealType>(1.0);
    m_HalfDerivativeWeights[i] = static_cast<TRealType>(0.5);
    }
}
template <typename TInputImage, typename TRealType, typename TOutputImage>
void
ExtractRotationsFilter<TInputImage, TRealType, TOutputImage>
::SetDerivativeWeights(TRealType data[])
{
  m_UseImageSpacing = false;

  for (unsigned int i = 0; i < ImageDimension; ++i)
    {
    if (m_DerivativeWeights[i] != data[i])
      {
      this->Modified();
      m_DerivativeWeights[i] = data[i];
      m_HalfDerivativeWeights[i] = 0.5*data[i];
      }
    }
}

template <typename TInputImage, typename TRealType, typename TOutputImage>
void
ExtractRotationsFilter<TInputImage, TRealType, TOutputImage>
::SetUseImageSpacing(bool f)
{
  if (m_UseImageSpacing == f)
    {
    return;
    }

  // Only reset the weights if they were previously set to the image spacing,
  // otherwise, the user may have provided their own weightings.
  if (f == false && m_UseImageSpacing == true)
    {
    for (unsigned int i = 0; i < ImageDimension; ++i)
      {
      m_DerivativeWeights[i] = static_cast<TRealType>(1.0);
      m_HalfDerivativeWeights[i] = static_cast<TRealType>(0.5);
      }
    }

  m_UseImageSpacing = f;
  this->Modified();
}

template <typename TInputImage, typename TRealType, typename TOutputImage>
void
ExtractRotationsFilter<TInputImage, TRealType, TOutputImage>
::GenerateInputRequestedRegion() throw(InvalidRequestedRegionError)
{
  // call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  // get pointers to the input and output
  InputImagePointer  inputPtr =
    const_cast< InputImageType * >( this->GetInput());
  OutputImagePointer outputPtr = this->GetOutput();

  if ( !inputPtr || !outputPtr )
    {
    return;
    }

  // get a copy of the input requested region (should equal the output
  // requested region)
  typename TInputImage::RegionType inputRequestedRegion;
  inputRequestedRegion = inputPtr->GetRequestedRegion();

  // pad the input requested region by the operator radius
  inputRequestedRegion.PadByRadius( m_NeighborhoodRadius );

  // crop the input requested region at the input's largest possible region
  if ( inputRequestedRegion.Crop(inputPtr->GetLargestPossibleRegion()) )
    {
    inputPtr->SetRequestedRegion( inputRequestedRegion );
    return;
    }
  else
    {
    // Couldn't crop the region (requested region is outside the largest
    // possible region).  Throw an exception.

    // store what we tried to request (prior to trying to crop)
    inputPtr->SetRequestedRegion( inputRequestedRegion );

    // build an exception
    InvalidRequestedRegionError e(__FILE__, __LINE__);
    e.SetLocation(ITK_LOCATION);
    e.SetDescription("Requested region is (at least partially) outside the largest possible region.");
    e.SetDataObject(inputPtr);
    throw e;
    }
}

template< typename TInputImage, typename TRealType, typename TOutputImage >
void
ExtractRotationsFilter<TInputImage, TRealType, TOutputImage>
::BeforeThreadedGenerateData()
{
  Superclass::BeforeThreadedGenerateData();

  // Set the weights on the derivatives.
  // Are we using image spacing in the calculations?  If so we must update now
  // in case our input image has changed.
  if (m_UseImageSpacing == true)
    {

    for (unsigned int i = 0; i < ImageDimension; i++)
      {
      if (static_cast<TRealType>(this->GetInput()->GetSpacing()[i]) == 0.0)
        {
        itkExceptionMacro(<< "Image spacing in dimension " << i << " is zero.");
        }
      m_DerivativeWeights[i]
        = static_cast<TRealType>( 1.0 /
                                  static_cast<TRealType>(this->GetInput()->GetSpacing()[i]) );
      m_HalfDerivativeWeights[i]=0.5*m_DerivativeWeights[i];
      }
    }

  /** If the input needs casting to a real-valued vector type, create the
      appropriate image and set the m_RealValuedInputImage pointer to this
      image.  Otherwise just point to the input image. */
  if ( typeid( typename InputImageType::PixelType ) != typeid( RealVectorType ) )
    {
    typename VectorCastImageFilter<TInputImage, RealVectorImageType>::Pointer
      caster = VectorCastImageFilter<TInputImage, RealVectorImageType>::New();
    caster->SetInput(this->GetInput());
    caster->Update();
    m_RealValuedInputImage = caster->GetOutput();
    }
  else
    {
    m_RealValuedInputImage
      = dynamic_cast<const ImageBase<ImageDimension> *>(this->GetInput());
    }

}

template< typename TInputImage, typename TRealType, typename TOutputImage >
void
ExtractRotationsFilter< TInputImage, TRealType, TOutputImage >
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                       int threadId)
{

  ZeroFluxNeumannBoundaryCondition<RealVectorImageType> nbc;
  ConstNeighborhoodIteratorType bit;
  ImageRegionIterator<TOutputImage> it;

  // Find the data-set boundary "faces"
  typename NeighborhoodAlgorithm::ImageBoundaryFacesCalculator<RealVectorImageType>::
    FaceListType faceList;
  NeighborhoodAlgorithm::ImageBoundaryFacesCalculator<RealVectorImageType> bC;
  faceList = bC(dynamic_cast<const RealVectorImageType *>(m_RealValuedInputImage.GetPointer()),
                outputRegionForThread, m_NeighborhoodRadius);

  typename NeighborhoodAlgorithm::ImageBoundaryFacesCalculator<RealVectorImageType>::
    FaceListType::iterator fit;
  fit = faceList.begin();

  // Support progress methods/callbacks
  ProgressReporter progress(this, threadId, outputRegionForThread.GetNumberOfPixels());

  // Process each of the data set faces.  The iterator is reinitialized on each
  // face so that it can determine whether or not to check for boundary
  // conditions.
  for (fit=faceList.begin(); fit != faceList.end(); ++fit)
    {
    bit = ConstNeighborhoodIteratorType(m_NeighborhoodRadius,
                                        dynamic_cast<const RealVectorImageType *>(m_RealValuedInputImage.GetPointer()),
                                        *fit);
    it = ImageRegionIterator<TOutputImage>(this->GetOutput(), *fit);
    bit.OverrideBoundaryCondition(&nbc);
    bit.GoToBegin();

    while ( ! bit.IsAtEnd() )
      {
      //it.Set( static_cast<OutputPixelType>( this->EvaluateAtNeighborhood(bit) ) );
      it.Set( static_cast<OutputPixelType>( this->EvaluateAtNeighborhood(bit) ) );
      ++bit;
      ++it;
      progress.CompletedPixel();
      }
    }
}

template <typename TInputImage, typename TRealType, typename TOutputImage>
//vnl_matrix_fixed< TRealType, ExtractRotationsFilter< TInputImage, TRealType, TOutputImage >::ImageDimension, ExtractRotationsFilter< TInputImage, TRealType, TOutputImage >::VectorDimension >
vnl_matrix<double>
ExtractRotationsFilter< TInputImage, TRealType, TOutputImage >
::EvaluateAtNeighborhood(const ConstNeighborhoodIteratorType &it) const
{
  //vnl_matrix_fixed<TRealType,ImageDimension,VectorDimension> J;
  vnl_matrix<double> J(ImageDimension,VectorDimension);
  for (unsigned int i = 0; i < ImageDimension; ++i)
    {
    for (unsigned int j = 0; j < VectorDimension; ++j)
      {
      J[i][j] = m_HalfDerivativeWeights[i] * (it.GetNext(i)[j] - it.GetPrevious(i)[j]);
      }
      // add one on the diagonal to consider the warping and not only the deformation field
      J[i][i] += 1.0;
    }
  //return vnl_det(J);
  
  vnl_matrix<double> JJT = J * J.transpose();
  vnl_svd<double> svd(JJT);
  vnl_diag_matrix<double> inv_sqrtW = svd.W();
  inv_sqrtW.invert_in_place();
  //inv_sqrtW = inv_sqrtW.asMatrix().apply(sqrt);
  inv_sqrtW.asMatrix().apply(sqrt);
  vnl_matrix<double> U = svd.U();
  vnl_matrix<double> Vt = svd.V().transpose();
  vnl_matrix<double> R = U * inv_sqrtW * Vt;
  return R;
}

template <typename TInputImage, typename TRealType, typename TOutputImage>
void
ExtractRotationsFilter<TInputImage, TRealType, TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
{
  unsigned int i;
  Superclass::PrintSelf(os,indent);
  os << indent << "m_UseImageSpacing = "          << m_UseImageSpacing
     << std::endl;
  os << indent << "m_RequestedNumberOfThreads = " << m_RequestedNumberOfThreads
     << std::endl;
  os << indent << "m_DerivativeWeights = ";
  for (i = 0; i < ImageDimension; i++)
    { os << m_DerivativeWeights[i] << " "; }
  os << std::endl;
  os << indent << "m_HalfDerivativeWeights = ";
  for (i = 0; i < ImageDimension; i++)
    { os << m_HalfDerivativeWeights[i] << " "; }
  os << std::endl;
  os << indent << "m_NeighborhoodRadius = "          << m_NeighborhoodRadius
     << std::endl;
  os << indent << "m_RealValuedInputImage = "          << m_RealValuedInputImage.GetPointer()
     << std::endl;
}

} // end namespace itk

#endif
