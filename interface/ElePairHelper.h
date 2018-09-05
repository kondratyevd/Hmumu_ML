#ifndef ELE_PAIR_HELPER
#define ELE_PAIR_HELPER

#include "UfHMuMuCode/UFDiMuonsAnalyzer/interface/CommonIncludes.h"
#include "UfHMuMuCode/UFDiMuonsAnalyzer/interface/ElePairInfo.h"
#include "UfHMuMuCode/UFDiMuonsAnalyzer/interface/EleInfo.h"

typedef struct {
  TLorentzVector nom;
  // TLorentzVector up1;
  // TLorentzVector down1;
  // TLorentzVector up2;
  // TLorentzVector down2;
} pairVecSys_ele;

typedef struct {
  TLorentzVector nom;
  // TLorentzVector up;
  // TLorentzVector down;
} eleVecSys;

void FillElePairInfos( ElePairInfos& _pairInfos, const EleInfos _eleInfos );

bool pair_is_OS_ele( std::pair< bool, std::pair<int, int> > i,
		 std::pair< bool, std::pair<int, int> > j );

void FillElePairMasses( eleVecSys& ele1_vec, eleVecSys& ele2_vec, pairVecSys_ele& pair_vec, 
		       // double& massErr, 
           const double MASS_ELEON,
		       const EleInfo _ele1, const EleInfo _ele2,
		       const double _ele1_pt, const double _ele2_pt
         //   ,
		       // const double _ele1_ptErr, const double _ele2_ptErr 
           );

#endif  // #ifndef ELE_PAIR_HELPER