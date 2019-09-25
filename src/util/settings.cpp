/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/



#include "util/settings.h"
#include <boost/bind.hpp>


namespace dso
{
int pyrLevelsUsed = PYR_LEVELS;


/* Parameters controlling when KF's are taken */
float setting_keyframesPerSecond = 0;   // if !=0, takes a fixed number of KF per second.
bool setting_realTimeMaxKF = false;   // if true, takes as many KF's as possible (will break the system if the camera stays stationary)
float setting_maxShiftWeightT= 0.04f * (640+480);
float setting_maxShiftWeightR= 0.0f * (640+480);
float setting_maxShiftWeightRT= 0.02f * (640+480);
float setting_kfGlobalWeight = 1;   // general weight on threshold, the larger the more KF's are taken (e.g., 2 = double the amount of KF's).
float setting_maxAffineWeight= 2;


/* initial hessian values to fix unobservable dimensions / priors on affine lighting parameters.
 */
float setting_idepthFixPrior = 50*50;
float setting_idepthFixPriorMargFac = 600*600;
float setting_initialRotPrior = 1e11;
float setting_initialTransPrior = 1e10;
float setting_initialAffBPrior = 1e14;
float setting_initialAffAPrior = 1e14;
float setting_initialCalibHessian = 5e9;





/* some modes for solving the resulting linear system (e.g. orthogonalize wrt. unobservable dimensions) */
int setting_solverMode = SOLVER_FIX_LAMBDA | SOLVER_ORTHOGONALIZE_X_LATER;
double setting_solverModeDelta = 0.00001;
bool setting_forceAceptStep = true;



/* some thresholds on when to activate / marginalize points */
float setting_minIdepthH_act = 100;
float setting_minIdepthH_marg = 50;



float setting_desiredImmatureDensity = 1500; // immature points per frame
float setting_desiredPointDensity = 2000; // aimed total points in the active window.
float setting_minPointsRemaining = 0.05;  // marg a frame if less than X% points remain.
float setting_maxLogAffFacInWindow = 0.7; // marg a frame if factor between intensities to current frame is larger than 1/X or X.


int   setting_minFrames = 5; // min frames in window.
int   setting_maxFrames = 7; // max frames in window.
int   setting_minFrameAge = 1;
int   setting_maxOptIterations=6; // max GN iterations.
int   setting_minOptIterations=1; // min GN iterations.
float setting_thOptIterations=1.2; // factor on break threshold for GN iteration (larger = break earlier)





/* Outlier Threshold on photometric energy */
float setting_outlierTH = 12*12;					// higher -> less strict
float setting_outlierTHSumComponent = 50*50; 		// higher -> less strong gradient-based reweighting .




int setting_pattern = 8;						// point pattern used. DISABLED.
float setting_margWeightFac = 0.5*0.5;          // factor on hessian when marginalizing, to account for inaccurate linearization points.


/* when to re-track a frame */
float setting_reTrackThreshold = 1.5; // (larger = re-track more often)



/* require some minimum number of residuals for a point to become valid */
int   setting_minGoodActiveResForMarg=3;
int   setting_minGoodResForMarg=4;






// 0 = nothing.
// 1 = apply inv. response.
// 2 = apply inv. response & remove V.
int setting_photometricCalibration = 2;
bool setting_useExposure = true;
float setting_affineOptModeA = 1e12; //-1: fix. >=0: optimize (with prior, if > 0).
float setting_affineOptModeB = 1e8; //-1: fix. >=0: optimize (with prior, if > 0).

int setting_gammaWeightsPixelSelect = 1; // 1 = use original intensity for pixel selection; 0 = use gamma-corrected intensity.




float setting_huberTH = 9; // Huber Threshold





// parameters controlling adaptive energy threshold computation.
float setting_frameEnergyTHConstWeight = 0.5;
float setting_frameEnergyTHN = 0.7f;
float setting_frameEnergyTHFacMedian = 1.5;
float setting_overallEnergyTHWeight = 1;
float setting_coarseCutoffTH = 20;





// parameters controlling pixel selection
float setting_minGradHistCut = 0.5;
float setting_minGradHistAdd = 7;
float setting_gradDownweightPerLevel = 0.75;
bool  setting_selectDirectionDistribution = true;






/* settings controling initial immature point tracking */
float setting_maxPixSearch = 0.027; // max length of the ep. line segment searched during immature point tracking. relative to image resolution.
float setting_minTraceQuality = 3;
int setting_minTraceTestRadius = 2;
int setting_GNItsOnPointActivation = 3;
float setting_trace_stepsize = 1.0;				// stepsize for initial discrete search.
int setting_trace_GNIterations = 3;				// max # GN iterations
float setting_trace_GNThreshold = 0.1;				// GN stop after this stepsize.
float setting_trace_extraSlackOnTH = 1.2;			// for energy-based outlier check, be slightly more relaxed by this factor.
float setting_trace_slackInterval = 1.5;			// if pixel-interval is smaller than this, leave it be.
float setting_trace_minImprovementFactor = 2;		// if pixel-interval is smaller than this, leave it be.




// for benchmarking different undistortion settings
float benchmarkSetting_fxfyfac = 0;
int benchmarkSetting_width = 0;
int benchmarkSetting_height = 0;
float benchmark_varNoise = 0;
float benchmark_varBlurNoise = 0;
float benchmark_initializerSlackFactor = 1;
int benchmark_noiseGridsize = 3;


float freeDebugParam1 = 1;
float freeDebugParam2 = 1;
float freeDebugParam3 = 1;
float freeDebugParam4 = 1;
float freeDebugParam5 = 1;



bool disableReconfigure=false;
bool debugSaveImages = false;
bool multiThreading = true;
bool disableAllDisplay = false;
bool setting_onlyLogKFPoses = true;
bool setting_logStuff = true;



bool goStepByStep = false;


bool setting_render_displayCoarseTrackingFull=false;
bool setting_render_renderWindowFrames=true;
bool setting_render_plotTrackingFull = false;
bool setting_render_display3D = true;
bool setting_render_displayResidual = true;
bool setting_render_displayVideo = true;
bool setting_render_displayDepth = true;
bool setting_render_displaySelection = true;
bool setting_render_displaySaliency = true;
bool setting_render_displayKF = true;

bool setting_fullResetRequested = false;

bool setting_debugout_runquiet = false;

int sparsityFactor = 5;	// not actually a setting, only some legacy stuff for coarse initializer.


void handleKey(char k)
{
	char kkk = k;
	switch(kkk)
	{
	case 'd': case 'D':
		freeDebugParam5 = ((int)(freeDebugParam5+1))%10;
		printf("new freeDebugParam5: %f!\n", freeDebugParam5);
		break;
	case 's': case 'S':
		freeDebugParam5 = ((int)(freeDebugParam5-1+10))%10;
		printf("new freeDebugParam5: %f!\n", freeDebugParam5);
		break;
	}

}




int staticPattern[10][40][2] = {
		{{0,0}, 	  {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},	// .
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{0,-1},	  {-1,0},	   {0,0},	    {1,0},	     {0,1}, 	  {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},	// +
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{-1,-1},	  {1,1},	   {0,0},	    {-1,1},	     {1,-1}, 	  {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},	// x
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{-1,-1},	  {-1,0},	   {-1,1},		{-1,0},		 {0,0},		  {0,1},	   {1,-1},		{1,0},		 {1,1},       {-100,-100},	// full-tight
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{0,-2},	  {-1,-1},	   {1,-1},		{-2,0},		 {0,0},		  {2,0},	   {-1,1},		{1,1},		 {0,2},       {-100,-100},	// full-spread-9
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{0,-2},	  {-1,-1},	   {1,-1},		{-2,0},		 {0,0},		  {2,0},	   {-1,1},		{1,1},		 {0,2},       {-2,-2},   // full-spread-13
		 {-2,2},      {2,-2},      {2,2},       {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{-2,-2},     {-2,-1}, {-2,-0}, {-2,1}, {-2,2}, {-1,-2}, {-1,-1}, {-1,-0}, {-1,1}, {-1,2}, 										// full-25
		 {-0,-2},     {-0,-1}, {-0,-0}, {-0,1}, {-0,2}, {+1,-2}, {+1,-1}, {+1,-0}, {+1,1}, {+1,2},
		 {+2,-2}, 	  {+2,-1}, {+2,-0}, {+2,1}, {+2,2}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{0,-2},	  {-1,-1},	   {1,-1},		{-2,0},		 {0,0},		  {2,0},	   {-1,1},		{1,1},		 {0,2},       {-2,-2},   // full-spread-21
		 {-2,2},      {2,-2},      {2,2},       {-3,-1},     {-3,1},      {3,-1}, 	   {3,1},       {1,-3},      {-1,-3},     {1,3},
		 {-1,3},      {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{0,-2},	  {-1,-1},	   {1,-1},		{-2,0},		 {0,0},		  {2,0},	   {-1,1},		{0,2},		 {-100,-100}, {-100,-100},	// 8 for SSE efficiency
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100},
		 {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}, {-100,-100}},

		{{-4,-4},     {-4,-2}, {-4,-0}, {-4,2}, {-4,4}, {-2,-4}, {-2,-2}, {-2,-0}, {-2,2}, {-2,4}, 										// full-45-SPREAD
		 {-0,-4},     {-0,-2}, {-0,-0}, {-0,2}, {-0,4}, {+2,-4}, {+2,-2}, {+2,-0}, {+2,2}, {+2,4},
		 {+4,-4}, 	  {+4,-2}, {+4,-0}, {+4,2}, {+4,4}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200},
		 {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}, {-200,-200}},
};

int staticPatternNum[10] = {
		1,
		5,
		5,
		9,
		9,
		13,
		25,
		21,
		8,
		25
};

int staticPatternPadding[10] = {
		1,
		1,
		1,
		1,
		2,
		2,
		2,
		3,
		2,
		4
};

bool setting_constant_smoothing = false;
bool setting_average_smoothing = false;
bool setting_blob_smoothing = false;
bool setting_segmentation_smoothing = false;

int setting_minSaliencyHistAdd = 127;
float setting_minSaliencyMeanWeight = 1.0;

float setting_seg_weights_map[150] = {
	0.1,   // wall
	1.0,   // building
	0.1,   // sky
	0.1,   // floor
	1.0,   // tree
	0.3,   // ceiling
	1.0,   // road
	1.0,   // bed
	1.0,   // windowpane
	1.0,   // grass
	1.0,  // cabinet
	1.0,  // sidewalk
	1.0,  // person
	1.0,  // earth
	1.0,  // door
	1.0,  // table
	1.0,  // mountain
	1.0,  // plant
	1.0,  // curtain
	1.0,  // chair
	1.0,  // car
	1.0,  // water
	1.0,  // painting
	1.0,  // sofa
	1.0,  // shelf
	1.0,  // house
	1.0,  // sea
	1.0,  // mirror
	1.0,  // rug
	1.0,  // field
	1.0,  // armchair
	1.0,  // seat
	1.0,  // fence
	1.0,  // desk
	1.0,  // rock
	1.0,  // wardrobe
	1.0,  // lamp
	1.0,  // bathtub
	1.0,  // railing
	1.0,  // cushion
	1.0,  // base
	1.0,  // box
	1.0,  // column
	1.0,  // signboard
	1.0,  // chest of drawer
	1.0,  // counters
	1.0,  // sand
	1.0,  // sink
	1.0,  // skyscraper
	1.0,  // fireplace
	1.0,  // refrigerator
	1.0,  // grandstand
	1.0,  // path
	1.0,  // stairs
	1.0,  // runway
	1.0,  // case
	1.0,  // pool table
	1.0,  // pillow
	1.0,  // screen door
	1.0,  // stairway
	1.0,  // river
	1.0,  // bridge
	1.0,  // bookcase
	1.0,  // blind
	1.0,  // coffee table
	1.0,  // toilet
	1.0,  // flower
	1.0,  // book
	1.0,  // hill
	1.0,  // bench
	1.0,  // countertop
	1.0,  // stove
	1.0,  // palm
	1.0,  // kitchen island
	1.0,  // computer
	1.0,  // swivel chair
	1.0,  // boat
	1.0,  // bar
	1.0,  // arcade machine
	1.0,  // hovel
	1.0,  // bus
	1.0,  // towel
	0.3,  // light
	1.0,  // truck
	1.0,  // tower
	1.0,  // chandelier
	1.0,  // awning
	1.0,  // streetlight
	1.0,  // booth
	1.0,  // television
	1.0,  // airplane
	1.0,  // dirt track
	1.0,  // apparel
	1.0,  // pole
	1.0,  // land
	1.0,  // bannister
	1.0,  // escalator
	1.0,  // ottoman
	1.0,  // bottle
	1.0,  // buffet
	1.0, // poster
	1.0, // stage
	1.0, // van
	1.0, // ship
	1.0, // fountain
	1.0, // conveyer belt
	1.0, // canopy
	1.0, // washer
	1.0, // plaything
	1.0, // swimming pool
	1.0, // stool
	1.0, // barrel
	1.0, // basket
	1.0, // waterfall
	1.0, // tent
	1.0, // bag
	1.0, // minibike
	1.0, // cradle
	1.0, // oven
	1.0, // ball
	1.0, // food
	1.0, // step
	1.0, // tank
	1.0, // trade name
	1.0, // microwave
	1.0, // pot
	1.0, // animal
	1.0, // bicycle
	1.0, // lake
	1.0, // dishwasher
	1.0, // screen
	1.0, // blanket
	1.0, // sculpture
	1.0, // hood
	1.0, // sconce
	1.0, // vase
	1.0, // traffic light
	1.0, // tray
	1.0, // ashcan
	1.0, // fan
	1.0, // pier
	1.0, // crt screen
	1.0, // plate
	1.0, // monitor
	1.0, // bulletin board
	1.0, // shower
	1.0, // radiator
	1.0, // glass
	1.0, // clock
	1.0  // flag
};

int setting_patch_size = 32;

bool setting_visualize_selection = false;

}
