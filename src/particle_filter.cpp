/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <float.h>
#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "helper_functions.h"
#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    num_particles = 20;

    default_random_engine gen;
    double std_x = std[0];
    double std_y = std[1];
    double std_theta = std[2];

    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    for (int i = 0; i < num_particles; i++) {
        Particle p = {};
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1;
        particles.push_back(p);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    default_random_engine gen;
    for (vector<Particle>::iterator it = particles.begin(); it < particles.end(); it++) {
        double x_f, y_f;
        if (yaw_rate > 0) {
            x_f = it->x + (velocity / yaw_rate) * (sin(it->theta + yaw_rate * delta_t) - sin(it->theta));
            y_f = it->y + (velocity / yaw_rate) * (cos(it->theta) - cos(it->theta + yaw_rate * delta_t));
        } else {
            x_f = it->x + velocity * delta_t * cos(it->theta);
            y_f = it->y + velocity * delta_t * sin(it->theta);
        }
        
        double theta_f = it->theta + yaw_rate * delta_t;

        double std_x = std_pos[0];
        double std_y = std_pos[1];
        double std_theta = std_pos[2];

        normal_distribution<double> dist_x(x_f, std_x);
        normal_distribution<double> dist_y(y_f, std_y);
        normal_distribution<double> dist_theta(theta_f, std_theta);

        it->x = dist_x(gen);
        it->y = dist_y(gen);
        it->theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // Did not end up using this.
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
    for (vector<Particle>::iterator particleIt = particles.begin(); particleIt < particles.end(); particleIt++) {
      // Filter out the landmarks that are too far away.
      vector<Map::single_landmark_s> closeLandmarks;
      for (auto landmark : map_landmarks.landmark_list) {
        if (dist(landmark.x_f, landmark.y_f, particleIt->x, particleIt->y) <= sensor_range) {
          closeLandmarks.push_back(landmark);  
        }
      }
        
      // Find the closest landmark for each observation.
      particleIt->weight = 1;

      for (vector<LandmarkObs>::iterator obsIt = observations.begin(); obsIt < observations.end(); obsIt++) {
          double obsMapX = obsIt->x * cos(particleIt->theta) - obsIt->y * sin(particleIt->theta) + particleIt->x;
          double obsMapY = obsIt->x * sin(particleIt->theta) + obsIt->y * cos(particleIt->theta) + particleIt->y;

          Map::single_landmark_s closestLandmark;
          double smallest_dist = DBL_MAX;
          for (auto landmark : closeLandmarks) {
            double d = dist(landmark.x_f, landmark.y_f, obsMapX, obsMapY);
            if (d < smallest_dist) {
                closestLandmark = landmark;
                smallest_dist = d;
            }       
          }
          double std_x = std_landmark[0];
          double std_y = std_landmark[1];
          double p_xy = exp(-(
              pow(obsMapX - closestLandmark.x_f, 2) / (2 * pow(std_x, 2)) + 
              pow(obsMapY - closestLandmark.y_f, 2) / (2 * pow(std_y, 2))));
          p_xy /= (2 * M_PI * std_x * std_y);

          particleIt->weight *= p_xy;
      }
    }

    // Normalize
    double weightSum = 0;
    for (auto particle : particles) {
        weightSum += particle.weight;
    }
    for (auto& particle : particles) {
        particle.weight /= weightSum;
    }
}

void ParticleFilter::resample() {
    vector<double> weights;
    for (auto particle : particles) {
        weights.push_back(particle.weight);
    }
    discrete_distribution<> dist(weights.begin(), weights.end());
    vector<Particle> newParticles;
    default_random_engine gen;
    
    for (int i = 0; i < num_particles; i++) {
        Particle sample = particles[dist(gen)];
        newParticles.push_back(sample);
    }
    
    particles = newParticles;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
