
#ifndef EVENT_HELPER
#define EVENT_HELPER

#include "UfHMuMuCode/UFDiMuonsAnalyzer/interface/CommonIncludes.h"
#include "UfHMuMuCode/UFDiMuonsAnalyzer/interface/EventInfo.h"

void FillEventInfo( EventInfo& _eventInfo, const edm::Event& iEvent );

#endif  // #ifndef EVENT_HELPER
