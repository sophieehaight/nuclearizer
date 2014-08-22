/*
 * MNCTStripHit.cxx
 *
 *
 * Copyright (C) by Andreas Zoglauer.
 * All rights reserved.
 *
 *
 * This code implementation is the intellectual property of
 * Andreas Zoglauer.
 *
 * By copying, distributing or modifying the Program (or any work
 * based on the Program) you indicate your acceptance of this statement,
 * and all its terms.
 *
 */


////////////////////////////////////////////////////////////////////////////////
//
// MNCTStripHit
//
////////////////////////////////////////////////////////////////////////////////


// Include the header:
#include "MNCTStripHit.h"

// Standard libs:
#include <iomanip>
using namespace std;

// ROOT libs:

// MEGAlib libs:
#include "MStreams.h"

////////////////////////////////////////////////////////////////////////////////


#ifdef ___CINT___
ClassImp(MNCTStripHit)
#endif


////////////////////////////////////////////////////////////////////////////////


MNCTStripHit::MNCTStripHit()
{
  // Construct an instance of MNCTStripHit

  m_ReadOutElement = new MReadOutElementDoubleStrip();
  
  Clear();
}


////////////////////////////////////////////////////////////////////////////////


MNCTStripHit::~MNCTStripHit()
{
  // Delete this instance of MNCTStripHit
  
  delete m_ReadOutElement;
}


////////////////////////////////////////////////////////////////////////////////


void MNCTStripHit::Clear()
{
  // Reset all data

  m_ReadOutElement->Clear();
  m_HasTriggered = false;
  m_UncorrectedADCUnits = 0;
  m_ADCUnits = 0;
  m_Energy = 0;
  m_EnergyResolution = 0;
  m_Timing = 0;
}


////////////////////////////////////////////////////////////////////////////////


bool MNCTStripHit::Parse(MString& Line, int Version)
{
  // to be written later 
  
  return  true;
}


////////////////////////////////////////////////////////////////////////////////


bool MNCTStripHit::StreamDat(ostream& S, int Version)
{
  //! Stream the content to an ASCII file 
  
  S<<"SH "
   <<m_ReadOutElement->GetDetectorID()<<" "
   <<((m_ReadOutElement->IsPositiveStrip() == true) ? "p" : "n")<<" "
   <<m_ReadOutElement->GetStripID()<<" "
   <<m_HasTriggered<<" "
   <<setprecision(9)<<m_Timing<<" "
   <<m_UncorrectedADCUnits<<" "
   <<m_ADCUnits<<" "
   <<m_Energy<<" "
   <<m_EnergyResolution<<endl;
 
  return true;
}


////////////////////////////////////////////////////////////////////////////////


void MNCTStripHit::StreamRoa(ostream& S)
{
  //! Stream the content in MEGAlib's evta format 

  S<<"UH " 
   <<m_ReadOutElement->GetDetectorID()<<" "
   <<m_ReadOutElement->GetStripID()<<" "
   <<((m_ReadOutElement->IsPositiveStrip() == true) ? "p" : "n")<<" "
   <<m_ADCUnits<<" "
   <<m_Timing<<endl;
}


// MNCTStripHit.cxx: the end...
////////////////////////////////////////////////////////////////////////////////
