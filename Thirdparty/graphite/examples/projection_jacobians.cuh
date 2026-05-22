#pragma once
template <typename Scalar>
__host__ __device__ void projection_simple(
    const Eigen::Matrix<Scalar, 3, 1> &rvec,
    const Eigen::Matrix<Scalar, 3, 1> &t, const Scalar f, const Scalar k1,
    const Scalar k2, const Eigen::Matrix<Scalar, 3, 1> &point,
    Eigen::Matrix<Scalar, 2, 9> &Jc, Eigen::Matrix<Scalar, 2, 3> &Jp) {
  // Operation counts:
  // add: 140
  // branch: 1
  // call: 3
  // compare: 1
  // divide: 7
  // multiply: 344
  // negate: 18
  // total: 514

  const Scalar v00009 = rvec(1, 0);
  const Scalar v00007 = rvec(0, 0);
  const Scalar v01484 = v00009 * v00009;
  const Scalar v01482 = v00007 * v00007;
  const Scalar v00011 = rvec(2, 0);
  const Scalar v01651 = v01482 + v01484;
  const Scalar v01488 = v00011 * v00011;
  const Scalar v00013 = v01488 + v01651;
  const Scalar v00014 = std::sqrt(v00013);
  const Scalar v01654 = -static_cast<Scalar>(2);
  Scalar v00033;
  Scalar v00039;
  Scalar v00045;
  Scalar v00052;
  Scalar v00057;
  Scalar v00061;
  Scalar v00069;
  Scalar v00072;
  Scalar v00076;
  Scalar v00102;
  Scalar v00111;
  Scalar v00119;
  Scalar v00126;
  Scalar v00135;
  Scalar v00141;
  Scalar v00166;
  Scalar v00169;
  Scalar v00176;
  Scalar v00193;
  Scalar v00198;
  Scalar v00206;
  Scalar v00213;
  Scalar v00222;
  Scalar v00228;
  Scalar v00239;
  Scalar v00242;
  Scalar v00245;
  Scalar v00258;
  Scalar v00263;
  Scalar v00270;
  Scalar v00278;
  Scalar v00284;
  Scalar v00287;
  Scalar v00298;
  Scalar v00301;
  Scalar v00304;
  if (static_cast<Scalar>(0) < v00014) {
    const Scalar v00022 = v00014 * static_cast<Scalar>(0.5);
    const Scalar v00089 = static_cast<Scalar>(1) / (v00013 * v00013);
    const Scalar v01656 = -v00007;
    const Scalar v01449 = v00009 * v00011;
    const Scalar v00023 = std::cos(v00022);
    const Scalar v01436 = v00089 * static_cast<Scalar>(4);
    const Scalar v00028 = static_cast<Scalar>(1) / v00013;
    const Scalar v00639 = v01449 + v01656;
    const Scalar v00096 = static_cast<Scalar>(1) / (v00014 * v00014 * v00014);
    const Scalar v00024 = std::sin(v00022);
    const Scalar v01448 = v00007 * v00011;
    const Scalar v01657 = -v00009;
    const Scalar v01447 = v00007 * v00009;
    const Scalar v01658 = -v00011;
    const Scalar v01659 = -v00023;
    const Scalar v01639 = v01436 * v01449;
    const Scalar v01424 = v00007 * v00028;
    const Scalar v01430 = v00024 * v00024;
    const Scalar v01419 = v00009 * v00028;
    const Scalar v01485 = v00009 * v00096;
    const Scalar v00534 = v01448 + v01657;
    const Scalar v01420 = v00011 * v00028;
    const Scalar v01454 = v00011 * v00096;
    const Scalar v00562 = v01447 + v01658;
    const Scalar v01409 = v00023 * static_cast<Scalar>(2);
    const Scalar v01423 = v00007 * v00024;
    const Scalar v00019 = static_cast<Scalar>(1) / v00014;
    const Scalar v01455 = v00023 * v00023;
    const Scalar v01652 = v01482 + v01488;
    const Scalar v01476 = static_cast<Scalar>(2) * v00028;
    const Scalar v01653 = v01484 + v01488;
    const Scalar v01425 = v00009 * v00024;
    const Scalar v00584 = v00007 + v01449;
    const Scalar v01426 = v00011 * v00024;
    const Scalar v01582 = v01409 * (v00024 * v00096);
    const Scalar v00485 = v00009 + v01448;
    const Scalar v00482 = v00023 * v01659 + v01430;
    const Scalar v01666 = -v00028;
    const Scalar v00538 = v01455 + -v01430;
    const Scalar v01099 = v00009 * v01436 * v01657 + v01476;
    const Scalar v01587 = v01409 * v01454;
    const Scalar v01166 = v00011 * v01436 * v01658 + v01476;
    const Scalar v01441 = v00019 * v00023;
    const Scalar v01428 = v00011 * v00023;
    const Scalar v01586 = (v00023 * v00096) * v01654;
    const Scalar v01457 = v00024 * static_cast<Scalar>(4);
    const Scalar v01101 = v01099 * v01426;
    const Scalar v01168 = v01166 * v01423;
    const Scalar v01173 = v01166 * v01425;
    const Scalar v00420 = v01420 * v01423;
    const Scalar v00424 = v01419 * v01426;
    const Scalar v01642 = v01430 * (v00028 * v01654);
    const Scalar v00438 = v01419 * v01423;
    const Scalar v01013 = v00482 * v01419 + v00485 * v01582;
    const Scalar v01521 = v00024 * (v00007 * v01436 * v01656 + v01476);
    const Scalar v01045 = v00482 * v01420 + (v00011 + v01447) * v01582;
    const Scalar v01055 = v00562 * v01582 + v00538 * v01420;
    const Scalar v01510 = v00007 * v00538;
    const Scalar v01483 = v00007 * v00482;
    const Scalar v01417 = v00024 * static_cast<Scalar>(2);
    const Scalar v00479 = v01426 * v01521;
    const Scalar v01037 = (v00089 * v01651 + v01666) * v01457 + v01586 * v01651;
    const Scalar v01327 = v00024 * v01436 + v01586;
    const Scalar v00519 = v01425 * v01521;
    const Scalar v01065 = (v00089 * v01652 + v01666) * v01457 + v01586 * v01652;
    const Scalar v00582 = (v01423 * v01639 + v00019 * v01409) * -v00024;
    const Scalar v01117 = v01586 * v01653 + (v00089 * v01653 + v01666) * v01457;
    const Scalar v00665 = v01423 * (v00024 * v01099);
    v00033 = ((v00009 * v00019) * v01659 + v00420) * v01417;
    v00039 = (v00007 * v01441 + v00424) * v01417;
    v00045 = static_cast<Scalar>(1) + v01642 * v01651;
    v00052 = static_cast<Scalar>(1) + v01642 * v01653;
    v00057 = (v01428 * -v00019 + v00438) * v01417;
    v00061 = (v00420 + v00009 * v01441) * v01417;
    v00069 = (v00438 + v00019 * v01428) * v01417;
    v00072 = static_cast<Scalar>(1) + v01642 * v01652;
    v00076 = (v00424 + (v00007 * v00019) * v01659) * v01417;
    v00102 = v00479 + v00007 * v01013;
    v00111 = v00007 * v01424 * v01455 +
             v00024 * ((v00019 + v00007 * (v00096 * v00639)) * v01409 +
                       -((v01424 + v01639) * v01423));
    v00119 = v01037 * v01423;
    v00126 = v01327 * v01423 * v01653;
    v00135 = v00519 + v00007 * v01045;
    v00141 = v00479 + v00007 * (v00534 * v01582 + v00538 * v01419);
    v00166 = v00519 + v00007 * v01055;
    v00169 = v01065 * v01423;
    v00176 = v00582 + v00007 * (v00584 * v01582 + v00482 * v01424);
    v00193 = v00582 + v00009 * v01013;
    v00198 = v01419 * v01510 + v00024 * (v01101 + v01485 * (v00639 * v01409));
    v00206 = v01037 * v01425;
    v00213 = v01117 * v01425;
    v00222 = v00665 + v00009 * v01045;
    v00228 = v00023 * v01419 * (v00009 * v00023) +
             v00024 * ((v00019 + v00534 * v01485) * v01409 +
                       -((v01436 * v01448 + v01419) * v01425));
    v00239 = v00665 + v00009 * v01055;
    v00242 = v01327 * v01425 * v01652;
    v00245 = v01419 * v01483 + v00024 * (v01101 + v01485 * (v00584 * v01409));
    v00258 = v01419 * (v00011 * v00482) + v00024 * (v01168 + v00485 * v01587);
    v00263 = v01420 * v01510 + v00024 * (v01173 + v00639 * v01587);
    v00270 = v01327 * v01426 * v01651;
    v00278 = v01117 * v01426;
    v00284 = v00582 + v00011 * v01045;
    v00287 = v01419 * (v00011 * v00538) + v00024 * (v01168 + v00534 * v01587);
    v00298 = v00023 * v01420 * v01428 +
             v00024 * ((v00019 + v00562 * v01454) * v01409 +
                       -((v01436 * v01447 + v01420) * v01426));
    v00301 = v01065 * v01426;
    v00304 = v01420 * v01483 + v00024 * (v01173 + v00584 * v01587);
  } else {
    v00033 = static_cast<Scalar>(0);
    v00039 = static_cast<Scalar>(0);
    v00045 = static_cast<Scalar>(1);
    v00052 = static_cast<Scalar>(1);
    v00057 = static_cast<Scalar>(0);
    v00061 = static_cast<Scalar>(0);
    v00069 = static_cast<Scalar>(0);
    v00072 = static_cast<Scalar>(1);
    v00076 = static_cast<Scalar>(0);
    v00102 = static_cast<Scalar>(0);
    v00111 = static_cast<Scalar>(0);
    v00119 = static_cast<Scalar>(0);
    v00126 = static_cast<Scalar>(0);
    v00135 = static_cast<Scalar>(0);
    v00141 = static_cast<Scalar>(0);
    v00166 = static_cast<Scalar>(0);
    v00169 = static_cast<Scalar>(0);
    v00176 = static_cast<Scalar>(0);
    v00193 = static_cast<Scalar>(0);
    v00198 = static_cast<Scalar>(0);
    v00206 = static_cast<Scalar>(0);
    v00213 = static_cast<Scalar>(0);
    v00222 = static_cast<Scalar>(0);
    v00228 = static_cast<Scalar>(0);
    v00239 = static_cast<Scalar>(0);
    v00242 = static_cast<Scalar>(0);
    v00245 = static_cast<Scalar>(0);
    v00258 = static_cast<Scalar>(0);
    v00263 = static_cast<Scalar>(0);
    v00270 = static_cast<Scalar>(0);
    v00278 = static_cast<Scalar>(0);
    v00284 = static_cast<Scalar>(0);
    v00287 = static_cast<Scalar>(0);
    v00298 = static_cast<Scalar>(0);
    v00301 = static_cast<Scalar>(0);
    v00304 = static_cast<Scalar>(0);
  }
  const Scalar v00041 = point(2, 0);
  const Scalar v00035 = point(1, 0);
  const Scalar v00005 = point(0, 0);
  const Scalar v00078 = t(1, 0);
  const Scalar v00063 = t(0, 0);
  const Scalar v00004 = t(2, 0);
  const Scalar v00466 =
      v00078 + v00005 * v00069 + v00035 * v00072 + v00041 * v00076;
  const Scalar v00448 =
      v00063 + v00005 * v00052 + v00035 * v00057 + v00041 * v00061;
  const Scalar v00047 =
      v00004 + v00005 * v00033 + v00035 * v00039 + v00041 * v00045;
  const Scalar v01554 = v00047 * v00047;
  const Scalar v01319 = v00448 * v00448 + v00466 * v00466;
  const Scalar v00049 = static_cast<Scalar>(1) / v01554;
  const Scalar v00084 = k2;
  const Scalar v01674 = -v01319;
  const Scalar v01671 = -v00049;
  const Scalar v00852 = v00005 * v00298 + v00035 * v00301 + v00041 * v00304;
  const Scalar v00816 = v00005 * v00278 + v00035 * v00284 + v00041 * v00287;
  const Scalar v00148 = static_cast<Scalar>(1) / (v00047 * v01554);
  const Scalar v01629 = v01671 * v01674;
  const Scalar v00725 = v00005 * v00239 + v00035 * v00242 + v00041 * v00245;
  const Scalar v00686 = v00005 * v00213 + v00035 * v00222 + v00041 * v00228;
  const Scalar v00594 = v00005 * v00166 + v00035 * v00169 + v00041 * v00176;
  const Scalar v00544 = v00005 * v00126 + v00035 * v00135 + v00041 * v00141;
  const Scalar v00003 = k1;
  const Scalar v01446 = v00148 * v01319;
  const Scalar v01408 = static_cast<Scalar>(-1) * static_cast<Scalar>(-1);
  const Scalar v00610 = v00003 + (v00084 * v01319) * v01654 * v01671;
  const Scalar v01599 = v01408 * v01446;
  const Scalar v00272 = v00005 * v00258 + v00035 * v00263 + v00041 * v00270;
  const Scalar v00208 = v00005 * v00193 + v00035 * v00198 + v00041 * v00206;
  const Scalar v00121 = v00005 * v00102 + v00035 * v00111 + v00041 * v00119;
  const Scalar v00144 = static_cast<Scalar>(1) / v00047;
  const Scalar v00001 = f;
  const Scalar v01598 = (v00466 * v00610) * v01654;
  const Scalar v01375 =
      v00272 * v01599 + (v00448 * v00816 + v00466 * v00852) * v01671;
  const Scalar v00475 =
      static_cast<Scalar>(1) + (v00003 + v00084 * v01629) * v01629;
  const Scalar v01359 =
      v00208 * v01599 + (v00448 * v00686 + v00466 * v00725) * v01671;
  const Scalar v01339 =
      v00121 * v01599 + (v00448 * v00544 + v00466 * v00594) * v01671;
  const Scalar v01597 = (v00448 * v00610) * v01654;
  const Scalar v01584 = v00466 * v01671;
  const Scalar v01590 = v00148 * v01674;
  const Scalar v01583 = v00448 * v01671;
  const Scalar v01655 = -v00001;
  const Scalar v01412 = static_cast<Scalar>(2) * v00610;
  const Scalar v01672 = -v00466;
  const Scalar v01581 = (v00144 * v00610) * v01654;
  const Scalar v01673 = -v00448;
  const Scalar v01585 = v01655 * v01671;
  const Scalar v01610 = v00466 * v01412;
  const Scalar v01593 = (v00466 * v00475) * v01671;
  const Scalar v01592 = (v00448 * v00475) * v01671;
  const Scalar v01641 = v01581 * v01672;
  const Scalar v01403 =
      v00045 * v01590 + v00049 * (v00061 * v00448 + v00076 * v00466);
  const Scalar v01399 =
      v00039 * v01590 + v00049 * (v00057 * v00448 + v00072 * v00466);
  const Scalar v01395 =
      v00033 * v01590 + v00049 * (v00052 * v00448 + v00069 * v00466);
  const Scalar v01640 = v01581 * v01673;
  const Scalar v01650 = (v01408 * v01671) * ((v01319 * v01319) * v01585);
  const Scalar v01646 = (v00144 * v01319) * v01585;
  const Scalar v01564 = v00144 * v00475;
  const Scalar v01617 =
      (v00610 * v01446 * v01654 * -v00144 + v00049 * v00475) * v01655;
  const Scalar v01606 = v00144 * v01655;
  const Scalar v00358 = (v00001 * v00144) * v01583 * v01610;
  Jc(0, 0) =
      (v00121 * v01592 + v00144 * (v00475 * v00544 + v01339 * v01597)) * v01655;
  Jc(0, 1) =
      (v00208 * v01592 + v00144 * (v00475 * v00686 + v01359 * v01597)) * v01655;
  Jc(0, 2) =
      (v00272 * v01592 + v00144 * (v00475 * v00816 + v01375 * v01597)) * v01655;
  Jc(0, 3) = (v00475 + (v00049 * v00448) * (v00448 * v01412)) * v01606;
  Jc(0, 4) = v00358;
  Jc(0, 5) = v01617 * v01673;
  Jc(0, 6) = v01564 * v01673;
  Jc(0, 7) = v01646 * v01673;
  Jc(0, 8) = (v00144 * v00448) * v01650;
  Jc(1, 0) =
      (v00121 * v01593 + v00144 * (v00475 * v00594 + v01339 * v01598)) * v01655;
  Jc(1, 1) =
      (v00208 * v01593 + v00144 * (v00475 * v00725 + v01359 * v01598)) * v01655;
  Jc(1, 2) =
      (v00272 * v01593 + v00144 * (v00475 * v00852 + v01375 * v01598)) * v01655;
  Jc(1, 3) = v00358;
  Jc(1, 4) = (v00475 + (v00049 * v00466) * v01610) * v01606;
  Jc(1, 5) = v01617 * v01672;
  Jc(1, 6) = v01564 * v01672;
  Jc(1, 7) = v01646 * v01672;
  Jc(1, 8) = (v00144 * v00466) * v01650;
  Jp(0, 0) =
      (v01395 * v01640 + v00475 * (v00033 * v01583 + v00052 * v00144)) * v01655;
  Jp(0, 1) =
      (v01399 * v01640 + v00475 * (v00039 * v01583 + v00057 * v00144)) * v01655;
  Jp(0, 2) =
      (v01403 * v01640 + v00475 * (v00045 * v01583 + v00061 * v00144)) * v01655;
  Jp(1, 0) =
      (v01395 * v01641 + v00475 * (v00033 * v01584 + v00069 * v00144)) * v01655;
  Jp(1, 1) =
      (v01399 * v01641 + v00475 * (v00039 * v01584 + v00072 * v00144)) * v01655;
  Jp(1, 2) =
      (v01403 * v01641 + v00475 * (v00045 * v01584 + v00076 * v00144)) * v01655;
}