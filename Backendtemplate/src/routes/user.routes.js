import express from 'express';
import { loginUser, logoutUser} from '../controllers/userControllers/userLogin.controller.js';
import { registerUser, } from '../controllers/userControllers/userRegister.controller.js';
import { verifyJWT } from '../middlewares/auth.middlewares.js';

const router = express.Router();


router.post('/register', registerUser);
router.post('/login', loginUser);
router.route('/logout').post(verifyJWT, logoutUser);

export default router;