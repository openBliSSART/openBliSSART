//
// $Id: audio.cpp 855 2009-06-09 16:15:50Z alex $
//
// This file is part of openBliSSART.
//
// Copyright (c) 2007-2009, Alexander Lehmann <lehmanna@in.tum.de>
//                          Felix Weninger <felix@weninger.de>
//                          Bjoern Schuller <schuller@tum.de>
//
// Institute for Human-Machine Communication
// Technische Universitaet Muenchen (TUM), D-80333 Munich, Germany
//
// openBliSSART is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 2 of the License, or (at your option) any later
// version.
//
// openBliSSART is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along with
// openBliSSART.  If not, see <http://www.gnu.org/licenses/>.
// 


#include <blissart/audio/audio.h>
#include <SDL/SDL.h>
#include <SDL/SDL_sound.h>
#include <cstdlib>


namespace blissart {

namespace audio {


static bool _sdlInitialized = false;
static bool _exitHookRegistered = false;


bool initialize()
{
    if (_sdlInitialized)
        return true;

    do {
        // First initialize the mothership.
        bool audioInitDone =
            ((SDL_WasInit(SDL_INIT_AUDIO) & SDL_INIT_AUDIO) == SDL_INIT_AUDIO);
        if (!audioInitDone && SDL_Init(SDL_INIT_AUDIO) != 0)
            break;
        // Then it's allied forces.
        if (!Sound_Init()) {
            SDL_Quit();
            break;
        }
        // We register shutdown() via atexit, so that even if the user forgets
        // about calling shutdown() explicitly, we still perform a nice
        // cleanup.
        if (!_exitHookRegistered) {
            _exitHookRegistered = (atexit(shutdown) == 0) ? 1 : 0;
        }
        // Done.
        _sdlInitialized = true;
        return true;
    } while (false);

    // Once we get to this point, an error has occured.
    return false;
}


void shutdown()
{
    if (!_sdlInitialized)
        return;
    // Shutdown happens in reverse order.
    Sound_Quit();
    SDL_Quit();
    _sdlInitialized = false;
}


bool isInitialized() {
    return _sdlInitialized;
}


} // namespace audio

} // namespace blissart

