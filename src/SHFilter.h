#include "itkBinaryFunctorImageFilter.h"
#include "SHBasis.h"
#include "vnl_qr.h"

namespace itk
{
namespace Function
{
    template<class TInput1, int L = 8>
    class ResampleDWIVoxel
    {
      private:
        vnl_vector<double> B;
        vnl_matrix<double> newY;
        unsigned int numberOfBaselineImages;
        bool with_baseline_images;

      public:

        ResampleDWIVoxel() 
        {
          with_baseline_images = false;
          B = ComputeB(L);
        }
        ~ResampleDWIVoxel() {}

        bool operator!=( const ResampleDWIVoxel & ) const
        {
          return false;
        }
        bool operator==( const ResampleDWIVoxel & other ) const
        {
          return !(*this != other);
        }

        void SetNewY(vnl_matrix<double> a_newY)
        {
          this->newY = a_newY;
        }
        void IncludeBaselineImagesInOutput()
        {
          this->with_baseline_images = true;
        }
        void ExcludeBaselineImagesFromOutput()
        {
          this->with_baseline_images = false;
        }

        typedef vnl_vector<double> VectorType;
        typedef vnl_matrix<double> MatrixType;

        inline itk::VariableLengthVector<double> operator() (const TInput1 & A, const vnl_matrix<double> & gradient_matrix) const
        {
          itk::VariableLengthVector< double > data = static_cast< itk::VariableLengthVector<double> > (A); 
          vnl_matrix<double> duplicated_gradients = static_cast< vnl_matrix<double> > (gradient_matrix);

          const unsigned int numberOfBaselineImages = data.GetNumberOfElements() - duplicated_gradients.rows()/2;

          /* get S = [data(j,:) data(j,:)], the data vector with twice the size as the number of gradients */
          //const unsigned int numberOfBaselineImages = 8;
          VectorType S(2*data.GetNumberOfElements()-(2*numberOfBaselineImages)); 
          for (unsigned int i = numberOfBaselineImages; i < data.GetNumberOfElements(); i++) 
          {
            S(i-numberOfBaselineImages) = data.GetElement(i);
            S(i-2*numberOfBaselineImages+data.GetNumberOfElements()) = data.GetElement(i);
          }
          for (unsigned int i = 0; i < duplicated_gradients.rows(); i++)
            for (unsigned int j = 0; j < duplicated_gradients.cols(); j++)
              assert(duplicated_gradients(i,j) == duplicated_gradients(i,j));
          /* Get the value for each of the (L+1)*(L+2)/2 SH basis functions at each gradient direction */
          MatrixType Y = GetSHBasis<double>(duplicated_gradients, L);
          /* Perform part of the gradient SH projection computation */ 
          MatrixType Y_t = Y.transpose();
          MatrixType denominator = Y_t * Y;
          vnl_diag_matrix<double> diag =  vnl_diag_matrix<double>(0.003 * B);
          denominator = denominator +  diag;
          denominator = vnl_matrix_inverse<double>( denominator );
          //denominator = vnl_qr<double>( denominator ).inverse();
          /* Compute the SH projection, 'Cs', of this voxel's gradient function onto (L+1)(L+2)/2 basis functions. So 'Cs' is a vector of size (L+1)(L+2)/2. */
          VectorType Cs = denominator * Y_t * S;
          /* Compute the voxel's values at the new sample directions */
          VectorType sh_coef = newY * Cs;
          /* Save the new values to the output image */ 
          itk::VariableLengthVector<double> final_data;
          unsigned int start = 0;
          if (this->with_baseline_images)
            start = numberOfBaselineImages;
          final_data.SetSize(start + sh_coef.size());
          for (unsigned int i = 0; i < start; i++)
          {
            final_data[i] = data.GetElement(i);
          }
          for (unsigned int i = start; i < final_data.GetSize(); i++)
          {
            final_data[i] = sh_coef[i-start];
          }

          return final_data;
        }
    };
}

template< typename TInputImage1, typename TOutputImage = TInputImage1, int L=8>
  class ITK_EXPORT SHFilter:
    public
    BinaryFunctorImageFilter< TInputImage1, itk::Image< vnl_matrix<double>, 3> , TOutputImage,
    Function::ResampleDWIVoxel< ITK_TYPENAME TInputImage1::PixelType, L >   >
{
  public:
    typedef SHFilter Self;
    typedef BinaryFunctorImageFilter<
      TInputImage1, itk::Image< vnl_matrix<double>, 3 >, TOutputImage,
      Function::ResampleDWIVoxel< ITK_TYPENAME TInputImage1::PixelType > > Superclass;

    typedef SmartPointer< Self >       Pointer;
    typedef SmartPointer< const Self > ConstPointer;

    typedef typename Superclass::OutputImageType OutputImageType;

    itkNewMacro(Self);
    itkTypeMacro(SHFilter, BinaryFunctorImageFilter);

    void SetSamples(vnl_matrix<double> samples)
    {
      vnl_matrix<double> newY = GetSHBasis<double>(samples, L); 
      this->GetFunctor().SetNewY(newY);
    }
    
    void SetOutputLength(unsigned int num)
    {
      this->output_length = num;
    }

  protected:
    SHFilter() {}
    virtual ~SHFilter() {}
    void UpdateOutputInformation()
    {
      Superclass::UpdateOutputInformation();
      this->GetOutput(0)->SetVectorLength(output_length);
    }

  private:
    SHFilter(const Self &); //purposely not implemented
    void operator=(const Self &);        //purposely not implemented

    unsigned int output_length;
    unsigned int numberOfBaselineImages;
};
} //end namespace itk
