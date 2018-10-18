// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/***************************************************************************
 *
 * Copyright (c) 2015 Baidu.com, Inc. All Rights Reserved
 *
 **************************************************************************/

/**
 * @file include/gpc.h
 * @author huhan02(com@baidu.com)
 * @date 2015/12/18 13:52:10
 * @brief
 *
 * @modified by sunyipeng
 * @email sunyipeng@baidu.com
 * @date 2018/6/12
 **/

#ifndef PADDLE_FLUID_OPERATORS_DETECTION_GPC_H_  // GPC_H_
#define PADDLE_FLUID_OPERATORS_DETECTION_GPC_H_  // GPC_H_

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

namespace gpc {

typedef enum {  // Set operation type
  GPC_DIFF,     // Difference
  GPC_INT,      // Intersection
  GPC_XOR,      // Exclusive or
  GPC_UNION     // Union
} gpc_op;

typedef struct {  // Polygon vertex structure
  double x;       // Vertex x component
  double y;       // vertex y component
} gpc_vertex;

typedef struct {       // Vertex list structure
  int num_vertices;    // Number of vertices in list
  gpc_vertex *vertex;  // Vertex array pointer
} gpc_vertex_list;

typedef struct {             // Polygon set structure
  int num_contours;          // Number of contours in polygon
  int *hole;                 // Hole  external contour flags
  gpc_vertex_list *contour;  // Contour array pointer
} gpc_polygon;

typedef struct {           // Tristrip set structure
  int num_strips;          // Number of tristrips
  gpc_vertex_list *strip;  // Tristrip array pointer
} gpc_tristrip;

typedef enum { LEFT, RIGHT } gpc_left_right;

typedef enum { ABOVE, BELOW } gpc_above_below;

typedef enum { CLIP, SUBJ } gpc_clip_subj;

typedef enum {      /* Edge intersection classes         */
               NUL, /* Empty non-intersection            */
               EMX, /* External maximum                  */
               ELI, /* External left intermediate        */
               TED, /* Top edge                          */
               ERI, /* External right intermediate       */
               RED, /* Right edge                        */
               IMM, /* Internal maximum and minimum      */
               IMN, /* Internal minimum                  */
               EMN, /* External minimum                  */
               EMM, /* External maximum and minimum      */
               LED, /* Left edge                         */
               ILI, /* Internal left intermediate        */
               BED, /* Bottom edge                       */
               IRI, /* Internal right intermediate       */
               IMX, /* Internal maximum                  */
               FUL  /* Full non-intersection             */
} vertex_type;

typedef enum {     /* Horizontal edge states            */
               NH, /* No horizontal edge                */
               BH, /* Bottom horizontal edge            */
               TH  /* Top horizontal edge               */
} h_state;

typedef enum {              /* Edge bundle state                 */
               UNBUNDLED,   /* Isolated edge not within a bundle */
               BUNDLE_HEAD, /* Bundle head node                  */
               BUNDLE_TAIL  /* Passive bundle tail node          */
} bundle_state;

typedef struct v_shape { /* Internal vertex list datatype     */
  double x;              /* X coordinate component            */
  double y;              /* Y coordinate component            */
  struct v_shape *next;  /* Pointer to next vertex in list    */
} vertex_node;

typedef struct p_shape { /* Internal contour / tristrip type  */
  int active;            /* Active flag / vertex count        */
  int hole;              /* Hole / external contour flag      */
  vertex_node *v[2];     /* Left and right vertex list ptrs   */
  struct p_shape *next;  /* Pointer to next polygon contour   */
  struct p_shape *proxy; /* Pointer to actual structure used  */
} polygon_node;

typedef struct edge_shape {
  gpc_vertex vertex;             /* Piggy-backed contour vertex data  */
  gpc_vertex bot;                /* Edge lower (x, y) coordinate      */
  gpc_vertex top;                /* Edge upper (x, y) coordinate      */
  double xb;                     /* Scanbeam bottom x coordinate      */
  double xt;                     /* Scanbeam top x coordinate         */
  double dx;                     /* Change in x for a unit y increase */
  int type;                      /* Clip / subject edge flag          */
  int bundle[2][2];              /* Bundle edge flags                 */
  int bside[2];                  /* Bundle left / right indicators    */
  bundle_state bstate[2];        /* Edge bundle state                 */
  polygon_node *outp[2];         /* Output polygon / tristrip pointer */
  struct edge_shape *prev;       /* Previous edge in the AET          */
  struct edge_shape *next;       /* Next edge in the AET              */
  struct edge_shape *pred;       /* Edge connected at the lower end   */
  struct edge_shape *succ;       /* Edge connected at the upper end   */
  struct edge_shape *next_bound; /* Pointer to next bound in LMT      */
} edge_node;

inline bool gpc_eq(float a, float b) { return (fabs(a - b) <= 1e-6); }

inline bool gpc_prev_index(float a, float b) { return (fabs(a - b) <= 1e-6); }

inline int gpc_prev_index(int i, int n) { return ((i - 1 + n) % n); }

inline int gpc_next_index(int i, int n) { return ((i + 1) % n); }

inline int gpc_optimal(gpc_vertex *v, int i, int n) {
  return (v[(i + 1) % n].y != v[i].y || v[(i - 1 + n) % n].y != v[i].y);
}

inline int gpc_fwd_min(edge_node *v, int i, int n) {
  return (v[(i + 1) % n].vertex.y > v[i].vertex.y &&
          v[(i - 1 + n) % n].vertex.y >= v[i].vertex.y);
}

inline int gpc_not_fmax(edge_node *v, int i, int n) {
  return (v[(i + 1) % n].vertex.y > v[i].vertex.y);
}

inline int gpc_rev_min(edge_node *v, int i, int n) {
  return (v[(i + 1) % n].vertex.y >= v[i].vertex.y &&
          v[(i - 1 + n) % n].vertex.y > v[i].vertex.y);
}

inline int gpc_not_rmax(edge_node *v, int i, int n) {
  return (v[(i - 1 + n) % n].vertex.y > v[i].vertex.y);
}

// inline void gpc_p_edge(edge_node *d, edge_node *e, int p, double i, double j)
// {
inline void gpc_p_edge(edge_node *d, edge_node *e, int p) {
  d = e;
  do {
    d = d->prev;
  } while (!d->outp[p]);
  // i = d->bot.x + d->dx * (j - d->bot.y);
}

// inline void gpc_n_edge(edge_node *d, edge_node *e, int p, double i, double j)
// {
inline void gpc_n_edge(edge_node *d, edge_node *e, int p) {
  d = e;
  do {
    d = d->next;
  } while (!d->outp[p]);
  // i = d->bot.x + d->dx * (j - d->bot.y);
}

template <typename T>
void gpc_malloc(T *&p, int b, char *s) {
  if (b > 0) {
    p = (T *)malloc(b);

    if (!p) {
      fprintf(stderr, "gpc malloc failure: %s\n", s);
      exit(0);
    }
  } else {
    p = NULL;
  }
}
template <typename T>
void gpc_free(T *&p) {
  if (p) {
    free(p);
    p = NULL;
  }
}

/*
===========================================================================
                       Public Function Prototypes
===========================================================================
*/

void add_vertex(vertex_node **t, double x, double y);

void gpc_vertex_create(edge_node *e, int p, int s, double x, double y);

/*
void gpc_read_polygon(FILE *infile_ptr, int read_hole_flags,
                      gpc_polygon *polygon);

void gpc_write_polygon(FILE *outfile_ptr, int write_hole_flags,
                       gpc_polygon *polygon);
*/
void gpc_add_contour(gpc_polygon *polygon, gpc_vertex_list *contour, int hole);

void gpc_polygon_clip(gpc_op set_operation, gpc_polygon *subject_polygon,
                      gpc_polygon *clip_polygon, gpc_polygon *result_polygon);

void gpc_tristrip_clip(gpc_op set_operation, gpc_polygon *subject_polygon,
                       gpc_polygon *clip_polygon,
                       gpc_tristrip *result_tristrip);

void gpc_polygon_to_tristrip(gpc_polygon *polygon, gpc_tristrip *tristrip);

void gpc_free_polygon(gpc_polygon *polygon);

void gpc_free_tristrip(gpc_tristrip *tristrip);

}  // namespace gpc

#endif  // PADDLE_FLUID_OPERATORS_DETECTION_GPC_H_
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
