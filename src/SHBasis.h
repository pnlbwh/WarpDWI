template< class PixelType>
vnl_matrix<double> GetSHBasis3(vnl_matrix<double> samples, int L)
{
  int numcoeff = (L+1)*(L+2)/2;
  typedef vnl_matrix<double> MatrixType;
  MatrixType Y(samples.rows(), numcoeff);

  /* this is the makespharms(u, L) function in Yogesh's Matlab code (/home/yogesh/yogesh_pi/phd/dwmri/fODF_SH/makespharms.m) */
  typedef neurolib::SphericalHarmonicPolynomial<3> SphericalHarmonicPolynomialType;
  SphericalHarmonicPolynomialType *sphm = new SphericalHarmonicPolynomialType();
  for (unsigned int i = 0; i < samples.rows(); i++)
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
  //std::cout << "num rows of Y is: " << Y.rows() << std::endl;
  return Y;
}

vnl_vector<double> ComputeB(int L)
{
  unsigned int num_basis_functions = (L+1)*(L+2)/2;
  vnl_vector<double> r(num_basis_functions);
  vnl_vector<double> a(1);
  a(0) = 1;
  int end = 0;
  for (int l = 0; l <= L; l+=2)
  {
    a.set_size(2*l+1);
    a.fill(l);
    r.update(a, end); end += a.size(); 
  }
  //VectorType B = element_product(r, r+1);
  vnl_vector<double> B = element_product(r, r+1);
  B = element_product(B,B);
  return B;
}
