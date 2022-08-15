/**
 * 
 * Copyright (c) 2020, Jose-Luis Lisani, joseluis.lisani@uib.es
 * All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef FTC_NEW_HEADER
#define FTC_NEW_HEADER

#include <vector>
#include <algorithm>

std::vector<int> FTCsegmentation(float *histIn, int sizeIn, float eps, 
                            unsigned char circularhist);

std::vector<int> FTCsegmentation(float *histIn, int sizeIn, float eps, 
                            unsigned char circularhist, std::vector<int>& separators);

std::vector<int> FTCsegmentation(float *histIn, int sizeIn, int cost_thresh = 10000, float eps = 0.0001, 
                                 unsigned char circularhist = 0);

#endif
