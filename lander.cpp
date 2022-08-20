// Mars lander simulator
// Version 1.11
// Mechanical simulation functions
// Gabor Csanyi and Andrew Gee, August 2019

// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation, to make use of it
// for non-commercial purposes, provided that (a) its original authorship
// is acknowledged and (b) no modified versions of the source code are
// published. Restriction (b) is designed to protect the integrity of the
// exercise for future generations of students. The authors would be happy
// to receive any suggested    by private correspondence to
// ahg@eng.cam.ac.uk and gc121@eng.cam.ac.uk.
#define _USE_MATH_DEFINES
#include "lander.h"
#include "math.h"
#include <cmath>

void autopilot (void)
  // Autopilot to adjust the engine throttle, parachute and attitude control
{
  // INSERT YOUR CODE HERE
}

vector3d gravity (vector3d force_posi)
  // Calculate the gravity force at given postion.
  // force_posi is the position at which we calculate the gravitational force.
{
    vector3d F_G;
    double dist,tot_mass;
    dist = force_posi.abs();
    tot_mass = UNLOADED_LANDER_MASS + FUEL_DENSITY * (fuel * FUEL_CAPACITY);
    F_G = - GRAVITY * MARS_MASS * (tot_mass) * force_posi / pow(dist,3.0);
    return F_G;
}

vector3d drag (vector3d force_posi, vector3d force_vel)
  // Find the drag force given the position and velocity.
{
  double density, proj_area, vel_mag;
  vector3d F_D;
  vel_mag = force_vel.abs();


  proj_area = M_PI * pow(LANDER_SIZE,2.0);
  density = atmospheric_density(force_posi);
  F_D = - 0.5 * density * DRAG_COEF_LANDER * proj_area * vel_mag *force_vel;
  return F_D;
}



void numerical_dynamics (void)
  // This is the function that performs the numerical integration to update the
  // lander's pose. The time step is delta_t (global variable).
{
  // INSERT YOUR CODE HERE

  static vector3d previous_position;
  vector3d new_position;

  vector3d acceleration, tot_force;
  double tot_mass;
  tot_mass = UNLOADED_LANDER_MASS + FUEL_DENSITY * (fuel * FUEL_CAPACITY);


  if (simulation_time == 0.0) {

    // do an Euler update for the first iteration

    // i.e. new_position = .... (Euler update, using position and velocity)

    // velocity = .... (Euler update, using acceleration)  
    
    acceleration = (gravity(position) + drag(position,velocity))/ tot_mass;
    new_position = position + delta_t * velocity;
    velocity = velocity + acceleration * delta_t;

  } else {
    // do a Verlet update on all subsequent iterations

    // i.e. new_position = .... (Verlet update, using position and previous_position)

    // velocity = ... (Verlet update, using new_position and position)
    tot_force = gravity(position) + drag(position,velocity);
    new_position = 2 * position - previous_position + pow(delta_t,2.0) * tot_force / tot_mass;
    velocity = (new_position - position)/delta_t;
  }

  previous_position = position;

  position = new_position;



  // Here we can apply an autopilot to adjust the thrust, parachute and attitude
  if (autopilot_enabled) autopilot();

  // Here we can apply 3-axis stabilization to ensure the base is always pointing downwards
  if (stabilized_attitude) attitude_stabilization();
}

void initialize_simulation (void)
  // Lander pose initialization - selects one of 10 possible scenarios
{
  // The parameters to set are:
  // position - in Cartesian planetary coordinate system (m)
  // velocity - in Cartesian planetary coordinate system (m/s)
  // orientation - in lander coordinate system (xyz Euler angles, degrees)
  // delta_t - the simulation time step
  // boolean state variables - parachute_status, stabilized_attitude, autopilot_enabled
  // scenario_description - a descriptive string for the help screen

  scenario_description[0] = "circular orbit";
  scenario_description[1] = "descent from 10km";
  scenario_description[2] = "elliptical orbit, thrust changes orbital plane";
  scenario_description[3] = "polar launch at escape velocity (but drag prevents escape)";
  scenario_description[4] = "elliptical orbit that clips the atmosphere and decays";
  scenario_description[5] = "descent from 200km";
  scenario_description[6] = "";
  scenario_description[7] = "";
  scenario_description[8] = "";
  scenario_description[9] = "";

  switch (scenario) {

  case 0:
    // a circular equatorial orbit
    position = vector3d(1.2*MARS_RADIUS, 0.0, 0.0);
    velocity = vector3d(0.0, -3247.087385863725, 0.0);
    orientation = vector3d(0.0, 90.0, 0.0);
    delta_t = 0.1;
    parachute_status = NOT_DEPLOYED;
    stabilized_attitude = false;
    autopilot_enabled = false;
    break;

  case 1:
    // a descent from rest at 10km altitude
    position = vector3d(0.0, -(MARS_RADIUS + 10000.0), 0.0);
    velocity = vector3d(0.0, 0.0, 0.0);
    orientation = vector3d(0.0, 0.0, 90.0);
    delta_t = 0.1;
    parachute_status = NOT_DEPLOYED;
    stabilized_attitude = true;
    autopilot_enabled = false;
    break;

  case 2:
    // an elliptical polar orbit
    position = vector3d(0.0, 0.0, 1.2*MARS_RADIUS);
    velocity = vector3d(3500.0, 0.0, 0.0);
    orientation = vector3d(0.0, 0.0, 90.0);
    delta_t = 0.1;
    parachute_status = NOT_DEPLOYED;
    stabilized_attitude = false;
    autopilot_enabled = false;
    break;

  case 3:
    // polar surface launch at escape velocity (but drag prevents escape)
    position = vector3d(0.0, 0.0, MARS_RADIUS + LANDER_SIZE/2.0);
    velocity = vector3d(0.0, 0.0, 5027.0);
    orientation = vector3d(0.0, 0.0, 0.0);
    delta_t = 0.1;
    parachute_status = NOT_DEPLOYED;
    stabilized_attitude = false;
    autopilot_enabled = false;
    break;

  case 4:
    // an elliptical orbit that clips the atmosphere each time round, losing energy
    position = vector3d(0.0, 0.0, MARS_RADIUS + 100000.0);
    velocity = vector3d(4000.0, 0.0, 0.0);
    orientation = vector3d(0.0, 90.0, 0.0);
    delta_t = 0.1;
    parachute_status = NOT_DEPLOYED;
    stabilized_attitude = false;
    autopilot_enabled = false;
    break;

  case 5:
    // a descent from rest at the edge of the exosphere
    position = vector3d(0.0, -(MARS_RADIUS + EXOSPHERE), 0.0);
    velocity = vector3d(0.0, 0.0, 0.0);
    orientation = vector3d(0.0, 0.0, 90.0);
    delta_t = 0.1;
    parachute_status = NOT_DEPLOYED;
    stabilized_attitude = true;
    autopilot_enabled = false;
    break;

  case 6:
    break;

  case 7:
    break;

  case 8:
    break;

  case 9:
    break;

  }
}
