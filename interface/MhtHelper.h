
#ifndef MHT_HELPER
#define MHT_HELPER

#include "UfHMuMuCode/UFDiMuonsAnalyzer/interface/CommonIncludes.h"
#include "UfHMuMuCode/UFDiMuonsAnalyzer/interface/MhtInfo.h"
#include "UfHMuMuCode/UFDiMuonsAnalyzer/interface/MuonInfo.h"
#include "UfHMuMuCode/UFDiMuonsAnalyzer/interface/EleInfo.h"
#include "UfHMuMuCode/UFDiMuonsAnalyzer/interface/JetInfo.h"

void FillMhtInfo( MhtInfo& _mhtInfo, const MuonInfos _muonInfos, 
		  const EleInfos _eleInfos, const JetInfos _jetInfos );

#endif  // #ifndef MHT_HELPER
