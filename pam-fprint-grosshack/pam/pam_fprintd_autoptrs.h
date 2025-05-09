/*
 * pam_fprint: PAM module for fingerprint authentication through fprintd
 * Copyright (C) 2020 Marco Trevisan <marco.trevisan@canonical.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#pragma once

#include <stdlib.h>

/* Define auto-pointers functions, based on GLib Macros */

#define _CLEANUP_FUNC(func) __attribute__((cleanup (func)))

#define _PF_AUTOPTR_FUNC_NAME(TypeName) pf_autoptr_cleanup_ ## TypeName
#define _PF_AUTOPTR_TYPENAME(TypeName)  TypeName ## _pf_autoptr

#define PF_DEFINE_AUTOPTR_CLEANUP_FUNC(TypeName, cleanup) \
  typedef TypeName *_PF_AUTOPTR_TYPENAME (TypeName);    \
  static __attribute__((__unused__)) inline void        \
  _PF_AUTOPTR_FUNC_NAME (TypeName) (TypeName **_ptr)    \
  { if (_ptr) (cleanup) (*_ptr); }

#define PF_DEFINE_AUTO_CLEAN_FUNC(TypeName, cleanup)  \
  static __attribute__((__unused__)) inline void    \
  _PF_AUTOPTR_FUNC_NAME (TypeName) (TypeName *_ptr) \
  { cleanup (_ptr); }

static inline void
autoptr_cleanup_generic_free (void *p)
{
  void **pp = (void **) p;

  free (*pp);
}

#define pf_autofree _CLEANUP_FUNC (autoptr_cleanup_generic_free)
#define pf_autoptr(TypeName) \
  _CLEANUP_FUNC (_PF_AUTOPTR_FUNC_NAME (TypeName)) \
  _PF_AUTOPTR_TYPENAME (TypeName)
#define pf_auto(TypeName) \
  _CLEANUP_FUNC (_PF_AUTOPTR_FUNC_NAME (TypeName)) TypeName

PF_DEFINE_AUTOPTR_CLEANUP_FUNC (sd_bus, sd_bus_unref)
PF_DEFINE_AUTOPTR_CLEANUP_FUNC (sd_bus_message, sd_bus_message_unref)
PF_DEFINE_AUTOPTR_CLEANUP_FUNC (sd_bus_slot, sd_bus_slot_unref)

PF_DEFINE_AUTO_CLEAN_FUNC (sd_bus_error, sd_bus_error_free)
