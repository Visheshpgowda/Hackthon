import express from 'express';
import {loginUser,} from '../controllers/userControllers/userLogin.controller';
import { registerUser, } from '../controllers/userControllers/userRegister.controller';
import { verifyJWT } from '../middlewares/auth.middlewares';

const router = express.Router();


router.post('/register', registerUser);
router.post('/login', loginUser);
router.route('/logout').post(verifyJWT, logoutUser);

export default router;
