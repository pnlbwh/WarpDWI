PROJECT( WarpDWI )

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

FIND_PACKAGE(Slicer3 REQUIRED)

ENABLE_TESTING()
INCLUDE(CTest)

IF (Slicer3_FOUND)
  INCLUDE(${Slicer3_USE_FILE})
  slicer3_set_default_install_prefix_for_external_projects()
  INCLUDE_DIRECTORIES(
    ${Slicer3_Libs_INCLUDE_DIRS} 
    ${Slicer3_Base_INCLUDE_DIRS}
    )
  SET (CLP WarpDWI)
  SET (${CLP}_SOURCE ${CLP}.cxx)
  GENERATECLP(${CLP}_SOURCE ${CLP}.xml)
  ADD_EXECUTABLE(${CLP} ${${CLP}_SOURCE})
  TARGET_LINK_LIBRARIES (${CLP}
    ITKIO
    ITKCommon
    ITKStatistics
    MRML
    vtkTeem
    vtkCommon
    )
  ADD_TEST(${CLP}Test1 ${CLP} --help)
  ADD_TEST(${CLP}Test2 ${CLP} --version)
ELSE (Slicer3_FOUND)
  MESSAGE(WARNING "Need a Slicer3 installation to build against. Please set Slicer3_DIR to <SlicerInstallation>/Slicer3-Build")
ENDIF (Slicer3_FOUND)
