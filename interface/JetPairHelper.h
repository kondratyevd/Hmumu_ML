
#ifndef JET_PAIR_HELPER
#define JET_PAIR_HELPER

#include "UfHMuMuCode/UFDiMuonsAnalyzer/interface/CommonIncludes.h"
#include "UfHMuMuCode/UFDiMuonsAnalyzer/interface/JetPairInfo.h"
#include "UfHMuMuCode/UFDiMuonsAnalyzer/interface/JetInfo.h"

void FillJetPairInfos( JetPairInfos& _pairInfos, const JetInfos _jetInfos );

bool sort_jet_pairs_by_mass( std::pair< float, std::pair<int, int> > i,
                             std::pair< float, std::pair<int, int> > j );

#endif  // #ifndef JET_PAIR_HELPER
